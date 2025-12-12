
import os
import gc
import argparse
import logging
import collections
from datetime import datetime
import time

# from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import copy
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
# torch.manual_seed(487)

from models import *
import dataset
from models.resnet import BasicBlock, Bottleneck
from loops_mixstd import train_distill

from finetune import AverageMeter, validate, accuracy
from compute_flops import compute_MACs_params
from models.AdaptorWarp import AdaptorWarp
from models.BlockReinitWarp import BlockReinitWarp
from past_src.Grasp import GraSP, compute_importance_resnet
from models.resnet import load_rm_block_state_dict
from past_src.distill_data import DistillData
from past_src.generate_data import arg_parse


class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
                
    def state_dict(self):
        return self.module.state_dict()

import os
import gc
import argparse
import logging
import collections
from datetime import datetime
import time

# from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import copy
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
# torch.manual_seed(487)

from models import *
import dataset
from models.resnet import BasicBlock, Bottleneck
from loops_mixstd import train_distill

from finetune import AverageMeter, validate, accuracy
from compute_flops import compute_MACs_params
from models.AdaptorWarp import AdaptorWarp
from models.BlockReinitWarp import BlockReinitWarp
from past_src.Grasp import GraSP, compute_importance_resnet
from models.resnet import load_rm_block_state_dict
from past_src.distill_data import DistillData
from past_src.generate_data import arg_parse


def assert_finite(name, tensor):
    """
    Return True if tensor contains only finite values. Print info and return False otherwise.
    """
    try:
        if not torch.isfinite(tensor).all():
            # use nanmin/nanmax for numeric reporting (guarded)
            try:
                mn = torch.nanmin(tensor)
                mx = torch.nanmax(tensor)
            except Exception:
                mn, mx = float("nan"), float("nan")
            print(f"[NaN/Inf] {name} has non-finite values (min={mn}, max={mx})")
            return False
    except Exception:
        print(f"[assert_finite] could not check {name} (type={type(tensor)})")
        return False
    return True


def safe_to_device(x, device="cuda"):
    """Convert numpy or tensor to torch tensor on device."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


class OutputHook:
    """
    Forward_hook used to get the output of the intermediate layer.
    """

    def __init__(self):
        self.outputs = None

    def hook(self, m, i, output):
        """
        Output hook function.
        Args:
            m: module
            i: input
            output: output
        """
        self.outputs = output

    def pre_hook(self, m, i):
        """
        Pre-forward hook function.
        Args:
            m: module
            i: input
        """
        self.outputs = i[0]

    def clear(self):
        """
        Clear the output.
        """
        self.outputs = None


def compute_variance_loss(model, rm_blocks):
    variance_loss = 0.0
    for name, param in model.named_parameters():
        # Match block names (e.g., 'layer3.1.conv1.weight')
        for block in rm_blocks:
            if block in name and "conv" in name:
                var = torch.sum(param)
                variance_loss += abs(var)
                # param.grad *= 2
            # print(f"block and name {block}: {name}")
    return variance_loss


def print_block_BN_statistics(model):
    # Iterate over all modules and look for blocks (e.g., BasicBlock or Bottleneck)
    for name, module in model.named_modules():
        print(f"Block: {name}")
        # Now iterate over submodules in the block and find BN layers
        for subname, submodule in module.named_modules():
            if isinstance(submodule, nn.BatchNorm2d):
                aggregated_mean = submodule.weight.data.mean().item()
                aggregated_var = submodule.weight.data.var().item()
                print(f"  BN layer {subname}:")
                print(f"    Running Mean: {aggregated_mean}")
                print(f"    Running Var : {aggregated_var}")
        print("-" * 50)
        break


def get_batchnorm_gamma_values(model):
    """
    Extract BatchNorm gamma values (weight parameters) from each block in the ResNet model.

    Args:
        model: PyTorch ResNet model

    Returns:
        dict: Dictionary containing gamma values for each BatchNorm layer, organized by block
    """
    gamma_values = {}

    # Iterate through all named modules
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Extract the gamma values (weight parameter)
            gamma = module.weight.data.clone().cpu().numpy()

            # Store with the full module name
            gamma_values[name] = {
                "gamma_values": gamma,
                "mean_gamma": float(gamma.mean()),
                "std_gamma": float(gamma.std()),
                "min_gamma": float(gamma.min()),
                "max_gamma": float(gamma.max()),
                "num_channels": len(gamma),
            }

    return gamma_values


def print_batchnorm_gamma_summary(model):
    """
    Print a summary of BatchNorm gamma values for each block.

    Args:
        model: PyTorch ResNet model
    """
    gamma_values = get_batchnorm_gamma_values(model)

    print("=" * 80)
    print("BatchNorm Gamma Values Summary")
    print("=" * 80)

    # Group by layer for better organization
    layer_groups = {}
    for name, data in gamma_values.items():
        # Extract layer information from name
        if "layer" in name:
            parts = name.split(".")
            if len(parts) >= 2:
                layer_name = f"{parts[0]}.{parts[1]}"  # e.g., 'layer1.0'
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append((name, data))
        else:
            # Handle non-layer BatchNorm layers (e.g., conv1, bn1)
            if "conv1" not in layer_groups:
                layer_groups["conv1"] = []
            layer_groups["conv1"].append((name, data))

    # Print summary for each layer group
    for layer_name, bn_layers in sorted(layer_groups.items()):
        print(f"\n{layer_name.upper()}:")
        print("-" * 40)

        for bn_name, data in bn_layers:
            # Extract a shorter name for display
            short_name = bn_name.split(".")[-1] if "." in bn_name else bn_name
            print(f"  {short_name}:")
            print(f"    Mean: {data['mean_gamma']:.4f}")
            print(f"    Std:  {data['std_gamma']:.4f}")
            print(f"    Min:  {data['min_gamma']:.4f}")
            print(f"    Max:  {data['max_gamma']:.4f}")
            print(f"    Channels: {data['num_channels']}")

    # Print overall statistics
    all_gammas = np.concatenate(
        [data["gamma_values"] for data in gamma_values.values()]
    )
    print(f"\n" + "=" * 80)
    print("OVERALL STATISTICS:")
    print(f"Total BatchNorm layers: {len(gamma_values)}")
    print(f"Total channels: {len(all_gammas)}")
    print(f"Global mean gamma: {all_gammas.mean():.4f}")
    print(f"Global std gamma:  {all_gammas.std():.4f}")
    print(f"Global min gamma:  {all_gammas.min():.4f}")
    print(f"Global max gamma:  {all_gammas.max():.4f}")
    print("=" * 80)

    return gamma_values


def save_batchnorm_gamma_to_file(model, filename="batchnorm_gamma_values.pickle"):
    """
    Save BatchNorm gamma values to a pickle file for later analysis.

    Args:
        model: PyTorch ResNet model
        filename: Output filename for the pickle file

    Returns:
        str: Path to the saved file
    """
    gamma_values = get_batchnorm_gamma_values(model)

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(filename)
    filename_with_timestamp = f"{base_name}_{timestamp}{ext}"

    with open(filename_with_timestamp, "wb") as f:
        pickle.dump(gamma_values, f)

    print(f"BatchNorm gamma values saved to: {filename_with_timestamp}")
    return filename_with_timestamp


def analyze_batchnorm_gamma_distribution(model, save_plot=True):
    """
    Analyze and visualize the distribution of BatchNorm gamma values.

    Args:
        model: PyTorch ResNet model
        save_plot: Whether to save the plot to file

    Returns:
        dict: Analysis results
    """
    gamma_values = get_batchnorm_gamma_values(model)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("BatchNorm Gamma Values Distribution Analysis", fontsize=16)

    # Plot 1: Histogram of all gamma values
    all_gammas = np.concatenate(
        [data["gamma_values"] for data in gamma_values.values()]
    )
    axes[0, 0].hist(all_gammas, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Distribution of All Gamma Values")
    axes[0, 0].set_xlabel("Gamma Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Box plot by layer
    layer_names = []
    layer_gammas = []

    for name, data in gamma_values.items():
        if "layer" in name:
            parts = name.split(".")
            if len(parts) >= 2:
                layer_name = f"{parts[0]}.{parts[1]}"
                if layer_name not in layer_names:
                    layer_names.append(layer_name)
                    layer_gammas.append([])
                layer_idx = layer_names.index(layer_name)
                layer_gammas[layer_idx].extend(data["gamma_values"])

    if layer_gammas:
        axes[0, 1].boxplot(layer_gammas, labels=layer_names)
        axes[0, 1].set_title("Gamma Values by Layer")
        axes[0, 1].set_ylabel("Gamma Value")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean gamma values by layer
    layer_means = []
    for gammas in layer_gammas:
        layer_means.append(np.mean(gammas))

    if layer_means:
        axes[1, 0].bar(range(len(layer_names)), layer_means)
        axes[1, 0].set_title("Mean Gamma Values by Layer")
        axes[1, 0].set_xlabel("Layer")
        axes[1, 0].set_ylabel("Mean Gamma Value")
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels(layer_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Standard deviation of gamma values by layer
    layer_stds = []
    for gammas in layer_gammas:
        layer_stds.append(np.std(gammas))

    if layer_stds:
        axes[1, 1].bar(range(len(layer_names)), layer_stds)
        axes[1, 1].set_title("Standard Deviation of Gamma Values by Layer")
        axes[1, 1].set_xlabel("Layer")
        axes[1, 1].set_ylabel("Std Gamma Value")
        axes[1, 1].set_xticks(range(len(layer_names)))
        axes[1, 1].set_xticklabels(layer_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"batchnorm_gamma_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_filename}")

    plt.show()

    return {
        "layer_names": layer_names,
        "layer_gammas": layer_gammas,
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "overall_stats": {
            "mean": float(all_gammas.mean()),
            "std": float(all_gammas.std()),
            "min": float(all_gammas.min()),
            "max": float(all_gammas.max()),
        },
    }


def compute_channel_similarity(model, layer_name):
    """
    Compute cosine similarity between channels of a specific layer.
    Args:
        model: PyTorch model
        layer_name: Name of the layer to analyze
    Returns:
        similarity_matrix: Matrix of cosine similarities between channels
    """
    # Get the layer
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break

    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Get the weights
    if isinstance(layer, nn.Conv2d):
        # For Conv2d, reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
        weights = layer.weight.data.view(layer.out_channels, -1)
    else:
        raise ValueError(f"Layer {layer_name} is not a Conv2d layer")

    # Normalize the weights
    weights_norm = F.normalize(weights, p=2, dim=1)

    # Compute cosine similarity
    similarity_matrix = torch.mm(weights_norm, weights_norm.t())

    return similarity_matrix


def analyze_channel_similarities(model, layer_names):
    """
    Analyze channel similarities for multiple layers and save results.
    Args:
        model: PyTorch model
        layer_names: List of layer names to analyze
    """
    results = {}
    for layer_name in layer_names:
        print(f"\nAnalyzing layer: {layer_name}")
        similarity_matrix = compute_channel_similarity(model, layer_name)

        # Convert to numpy for easier analysis
        similarity_np = similarity_matrix.cpu().numpy()

        # Calculate statistics
        mean_similarity = np.mean(similarity_np)
        max_similarity = np.max(similarity_np)
        min_similarity = np.min(similarity_np)

        print(f"Mean similarity: {mean_similarity:.4f}")
        print(f"Max similarity: {max_similarity:.4f}")
        print(f"Min similarity: {min_similarity:.4f}")

        # Find most similar channel pairs
        np.fill_diagonal(similarity_np, -1)  # Exclude self-similarity
        max_sim_idx = np.unravel_index(np.argmax(similarity_np), similarity_np.shape)
        print(
            f"Most similar channels: {max_sim_idx} with similarity {similarity_np[max_sim_idx]:.4f}"
        )

        # Store results
        results[layer_name] = {
            "similarity_matrix": similarity_np,
            "mean_similarity": mean_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "most_similar_pair": (max_sim_idx, similarity_np[max_sim_idx]),
        }

        # Save similarity matrix
        save_path = f"channel_similarities_{layer_name.replace('.', '_')}.npy"
        np.save(save_path, similarity_np)
        print(f"Saved similarity matrix to {save_path}")

    return results


def get_block_features(model, block_name, data_loader, num_batches=10):
    """
    Extract input and output features of a specific block, capturing output after residual connection.
    Args:
        model: PyTorch model
        block_name: Name of the block to analyze (e.g., 'layer1.0')
        data_loader: DataLoader for input data
        num_batches: Number of batches to process
    Returns:
        input_features: Average input features
        output_features: Average output features (after residual)
    """
    model.eval()
    input_features = []
    output_features = []

    # Register hooks
    input_hook = OutputHook()
    output_hook = OutputHook()

    # Find the block
    block = None
    for name, module in model.named_modules():
        if name == block_name:
            block = module
            break

    if block is None:
        raise ValueError(f"Block {block_name} not found in model")

    # Register hooks with correct functions
    block.register_forward_pre_hook(lambda m, i: input_hook.pre_hook(m, i))
    # Register hook on the block itself to get output after residual
    block.register_forward_hook(lambda m, i, o: output_hook.hook(m, i, o))

    # Process batches
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.cuda()
            _ = model(data)

            # Store features
            if input_hook.outputs is not None:
                # Flatten spatial dimensions and average across batch
                input_feat = input_hook.outputs.view(input_hook.outputs.size(0), -1)
                input_features.append(input_feat.mean(dim=0).cpu())
            if output_hook.outputs is not None:
                # Flatten spatial dimensions and average across batch
                output_feat = output_hook.outputs.view(output_hook.outputs.size(0), -1)
                output_features.append(output_feat.mean(dim=0).cpu())

    # Average features across batches
    input_features = torch.stack(input_features).mean(dim=0)
    output_features = torch.stack(output_features).mean(dim=0)

    return input_features, output_features


def compute_block_similarities(model, target_block, data_loader, num_batches=10):
    """
    Compute similarities between target block and all other blocks.
    Args:
        model: PyTorch model
        target_block: Name of the target block (e.g., 'layer1.0')
        data_loader: DataLoader for input data
        num_batches: Number of batches to process
    Returns:
        similarities: Dictionary containing similarity matrices
    """
    # Get target block features
    target_input, target_output = get_block_features(
        model, target_block, data_loader, num_batches
    )

    # Get all block names (only BasicBlock or Bottleneck)
    block_names = []
    for name, module in model.named_modules():
        # Check if it's a block (BasicBlock or Bottleneck)
        if isinstance(module, (BasicBlock, Bottleneck)):
            # Extract the block name (e.g., 'layer1.0' from 'layer1.0.conv1')
            block_name = ".".join(name.split(".")[:2])
            if block_name not in block_names:
                block_names.append(block_name)

    similarities = {"input": {}, "output": {}}

    # Compute similarities with all other blocks
    for block_name in block_names:
        if block_name != target_block:
            block_input, block_output = get_block_features(
                model, block_name, data_loader, num_batches
            )

            # Ensure features have the same dimension
            min_dim = min(target_input.size(0), block_input.size(0))
            target_input_trimmed = target_input[:min_dim]
            block_input_trimmed = block_input[:min_dim]

            min_dim = min(target_output.size(0), block_output.size(0))
            target_output_trimmed = target_output[:min_dim]
            block_output_trimmed = block_output[:min_dim]

            # Compute cosine similarities
            input_sim = F.cosine_similarity(
                target_input_trimmed.unsqueeze(0), block_input_trimmed.unsqueeze(0)
            )
            output_sim = F.cosine_similarity(
                target_output_trimmed.unsqueeze(0), block_output_trimmed.unsqueeze(0)
            )

            similarities["input"][block_name] = input_sim.item()
            similarities["output"][block_name] = output_sim.item()

    return similarities


def plot_block_similarities(
    similarities, target_block, save_path="block_similarities.png"
):
    """
    Plot similarities between target block and all other blocks.
    Args:
        similarities: Dictionary containing similarity matrices
        target_block: Name of the target block
        save_path: Path to save the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Prepare data
    blocks = list(similarities["input"].keys())
    input_sims = [similarities["input"][b] for b in blocks]
    output_sims = [similarities["output"][b] for b in blocks]

    # Sort by input similarity
    sorted_idx = np.argsort(input_sims)
    blocks = [blocks[i] for i in sorted_idx]
    input_sims = [input_sims[i] for i in sorted_idx]
    output_sims = [output_sims[i] for i in sorted_idx]

    # Plot input similarities
    ax1.bar(range(len(blocks)), input_sims)
    ax1.set_title(
        f"Input Feature Similarities with Block {target_block}\n(Before Residual)"
    )
    ax1.set_xlabel("Blocks")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_xticks(range(len(blocks)))
    ax1.set_xticklabels(blocks, rotation=45, ha="right")

    # Plot output similarities
    ax2.bar(range(len(blocks)), output_sims)
    ax2.set_title(
        f"Output Feature Similarities with Block {target_block}\n(Before Residual)"
    )
    ax2.set_xlabel("Blocks")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_xticks(range(len(blocks)))
    ax2.set_xticklabels(blocks, rotation=45, ha="right")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Also save the raw similarity data
    np.save(
        save_path.replace(".png", "_data.npy"),
        {
            "blocks": blocks,
            "input_similarities": input_sims,
            "output_similarities": output_sims,
        },
    )


def compute_block_io_similarity(model, block_name, data_loader, num_batches=10):
    """
    Compute cosine similarity between input and output features of a specific block.
    Args:
        model: PyTorch model
        block_name: Name of the block to analyze
        data_loader: DataLoader for input data
        num_batches: Number of batches to process
    Returns:
        float: Average cosine similarity between input and output features
    """
    input_features, output_features = get_block_features(
        model, block_name, data_loader, num_batches
    )

    # Ensure features have the same dimension
    min_dim = min(input_features.size(0), output_features.size(0))
    input_features = input_features[:min_dim]
    output_features = output_features[:min_dim]

    # Compute cosine similarity
    similarity = F.cosine_similarity(
        input_features.unsqueeze(0), output_features.unsqueeze(0)
    )

    return similarity.item()


def analyze_block_io_similarities(
    model, data_loader, save_path="block_io_similarities.png", skip_blocks=None
):
    """
    Analyze and visualize input-output similarities for all blocks.
    Args:
        model: PyTorch model
        data_loader: DataLoader for input data
        save_path: Path to save the plot
        skip_blocks: List of block names to skip in analysis
    """
    # Get all block names
    block_names = []
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            block_name = ".".join(name.split(".")[:2])
            if block_name not in block_names:
                block_names.append(block_name)

    # Skip specified blocks
    if skip_blocks is not None:
        if isinstance(skip_blocks, str):
            skip_blocks = [skip_blocks]
        block_names = [name for name in block_names if name not in skip_blocks]

    # Compute similarities for each block
    similarities = {}
    for block_name in block_names:
        similarity = compute_block_io_similarity(model, block_name, data_loader)
        similarities[block_name] = similarity

    # Sort blocks by similarity
    sorted_blocks = sorted(similarities.items(), key=lambda x: x[1])
    block_names = [b[0] for b in sorted_blocks]
    sim_values = [b[1] for b in sorted_blocks]

    # Create plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(block_names)), sim_values)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.title(
        "Input-Output Feature Similarity for Each Block\n(After Residual Connection)"
    )
    plt.xlabel("Blocks")
    plt.ylabel("Cosine Similarity")
    plt.xticks(range(len(block_names)), block_names, rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Save raw data
    np.save(
        save_path.replace(".png", "_data.npy"),
        {"blocks": block_names, "similarities": sim_values},
    )

    return similarities


def compute_similarity(feature1, feature2):
    """
    Compute cosine similarity between two feature vectors.
    Args:
        feature1: First feature vector
        feature2: Second feature vector
    Returns:
        float: Average cosine similarity between input and output features
    """
    eps = 1e-8
    # Align batch size if needed
    N = min(feature1.size(0), feature2.size(0))
    f1 = feature1[:N]  # [N, D]
    f2 = feature2[:N]  # [N, D]

    # Normalize each vector
    f1_n = f1 / (f1.norm(dim=1, keepdim=True) + eps)
    f2_n = f2 / (f2.norm(dim=1, keepdim=True) + eps)

    # Per-sample cosine similarity: [N]
    cos_per_sample = (f1_n * f2_n).sum(dim=1)

    # Return average similarity
    return cos_per_sample.mean().item()


def Practise_one_block(
    rm_block, origin_model, origin_lat, train_loader, metric_loader, args, drop_blocks=0
):
    gc.collect()
    torch.cuda.empty_cache()

    pruned_model, _, pruned_lat = build_student(
        args.model,
        rm_block,
        args.num_classes,
        state_dict_path=args.state_dict_path,
        teacher=args.teacher,
        cuda=args.cuda,
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f"=> latency reduction: {lat_reduction:.2f}%")

    # print("Metric w/o Recovering:")
    # metric(metric_loader, pruned_model, origin_model)

    pruned_model_adaptor = AdaptorWarp(pruned_model)

    start_time = time.time()
    Practise_recover(train_loader, origin_model, pruned_model_adaptor, rm_block, args)
    print("Total time: {:.3f}s".format(time.time() - start_time))

    print("Metric w/ Recovering:")
    recoverability = metric(metric_loader, pruned_model_adaptor, origin_model)
    pruned_model_adaptor.remove_all_preconv()
    pruned_model_adaptor.remove_all_afterconv()

    # score = recoverability / lat_reduction
    score = 0
    # print(f"{rm_block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
    device = "cuda"
    DD = DistillData(args)
    dataloader = DD.get_distil_data(
        model_name=args.model,
        teacher_model=origin_model.cuda(),
        batch_size=args.batch_size,
        group=args.group,
        beta=0.1,
        gamma=0.5,
        save_path_head=args.save_path_head,
    )
    return pruned_model, (recoverability, lat_reduction, score)


def Practise_all_blocks(
    rm_blocks,
    origin_model,
    origin_lat,
    train_loader,
    metric_loader,
    args,
    drop_blocks=0,
):
    recoverabilities = dict()
    for rm_block in rm_blocks:
        _, results = Practise_one_block(
            rm_block, origin_model, origin_lat, train_loader, metric_loader, args
        )
        recoverabilities[rm_block] = results

    print("-" * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score = recoverabilities[block]
        print(f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
        sort_list.append([score, block])
    print("-" * 50)
    print("=> sorted")
    sort_list.sort()
    for score, block in sort_list:
        print(f"{block} -> {score:.4f}")
    print("-" * 50)
    print(f"=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})")
    print("Please use this seed to recover the model!")
    print("-" * 50)

    drop_blocks = []
    if args.rm_blocks.isdigit():
        for i in range(int(args.rm_blocks)):
            drop_blocks.append(sort_list[i][1])
    pruned_model, _, pruned_lat = build_student(
        args.model,
        drop_blocks,
        args.num_classes,
        state_dict_path=args.state_dict_path,
        teacher=args.teacher,
        cuda=args.cuda,
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f"=> latency reduction: {lat_reduction:.2f}%")

    return pruned_model, drop_blocks


# def Practise_one_block(
#     rm_block, origin_model, origin_lat, train_loader, metric_loader, args, drop_blocks=0
# ):
#     gc.collect()
#     torch.cuda.empty_cache()

#     if type(rm_block) is not list:
#         rm_block = [rm_block]

#     pruned_model, pruned_blocks, pruned_lat = build_student(
#         args.model,
#         rm_block,
#         args.num_classes,
#         state_dict_path=args.state_dict_path,
#         teacher=args.teacher,
#         cuda=args.cuda,
#     )

#     # example_batchnorm_gamma_analysis(origin_model)
#     # recoverability,acc,origin_acc,problematic_classes = metric(metric_loader, pruned_model, origin_model, trained=True)
#     # recoverability,pruned_acc,origin_acc,problematic_classes = metric(metric_loader, pruned_model, origin_model, trained=False)
#     pruned_acc = 0
#     pruned_model_adaptor = AdaptorWarp(pruned_model)
#     # pruned_model_adaptor = BlockReinitWarp(pruned_model)

#     start_time = time.time()
#     pruned_acc1 = Practise_recover(
#         train_loader, metric_loader, origin_model, pruned_model_adaptor, rm_block, args
#     )
#     print("Total time: {:.3f}s".format(time.time() - start_time))

#     print("Metric w/ Recovering:")
#     recoverability, acc, origin_acc, problematic_classes = metric(
#         metric_loader, pruned_model_adaptor, origin_model, trained=True
#     )

#     pruned_model_adaptor.remove_all_preconv()
#     pruned_model_adaptor.remove_all_afterconv()

#     # # After recoverability,acc,origin_acc,problematic_classes = metric(...)
#     # problematic_labels = problematic_classes  # problematic_classes from metric return
#     # with open('problematic_labels.pickle', 'wb') as f:
#     #     pickle.dump(problematic_labels, f)

#     # # Generate synthetic data using the pruned model with adaptors
#     # from past_src.distill_data import DistillData

#     # acc = 0
#     lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
#     print(f"=> latency reduction: {lat_reduction:.2f}%")
#     device = "cuda"
#     DD = DistillData(args)
#     dataloader = DD.get_distil_data(
#         model_name=args.model,
#         teacher_model=pruned_model_adaptor.cuda(),
#         batch_size=args.batch_size,
#         group=args.group,
#         beta=0.1,
#         gamma=0.5,
#         save_path_head=args.save_path_head,
#     )

#     # score = ((10-drop_blocks)/10)*(acc-origin_acc) + (drop_blocks/10)*(origin_acc-pruned_acc)
#     # score = GraSP(origin_model, 0.85, train_loader, 'cuda')
#     # score = compute_importance_resnet(origin_model, method='l1', use_cuda=True)
#     # for block, _ in score.items():
#     #     print(block, score[block])
#     score = 0
#     print(f"accuracy = {acc:0.4f}")
#     print(f"original accuracy = {origin_acc:0.4f}")
#     # print(f"score = {score:0.4f}")
#     # print(f"{rm_block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
#     recoverabilities = measure_taylor_saliency_per_block(origin_model, train_loader)
#     print(recoverabilities)

#     return pruned_model, (recoverability, lat_reduction, score)


# def Practise_all_blocks(
#     rm_blocks, origin_model, origin_lat, train_loader, metric_loader, args, drop_blocks
# ):
#     recoverabilities = dict()
#     # overall_mean = []
#     # print_block_BN_statistics(origin_model)
#     print(args.state_dict_path)
#     for rm_block in rm_blocks:
#         # load_rm_block_state_dict(pruned_model, raw_state_dict, rm_block, verbose=True)
#         drop_blocks.append(rm_block)
#         pruned_model, results = Practise_one_block(
#             drop_blocks,
#             origin_model,
#             origin_lat,
#             train_loader,
#             metric_loader,
#             args,
#             len(drop_blocks),
#         )
#         recoverabilities[rm_block] = results
#         # _,_,_,overall_mean = results
#         drop_blocks.remove(rm_block)

#     print("-" * 50)
#     sort_list = []
#     for block in recoverabilities:
#         recoverability, lat_reduction, score = recoverabilities[block]
#         print(f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
#         sort_list.append([score, block])
#     print("-" * 50)
#     print("=> sorted")
#     sort_list.sort(reverse=True)
#     # sort_list.sort()

#     for score, block in sort_list:
#         print(f"{block} -> {score:.4f}")
#     print("-" * 50)
#     print(f"=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})")
#     print("Please use this seed to recover the model!")
#     print("-" * 50)

#     # if args.rm_blocks.isdigit():
#     #     for i in range(int(args.rm_blocks)):
#     #         #print(sort_list[i][1]) #.remove(drop_blocks)
#     #         drop_blocks.append(sort_list[i][1])
#     # pruned_model, pruned_blocks, pruned_lat = build_student(
#     #     args.model, drop_blocks, args.num_classes,
#     #     state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
#     # )
#     # drop_blocks = []
#     # if args.rm_blocks.isdigit():
#     #     print(int(args.rm_blocks))
#     #     print(sort_list[0])
#     #     for i in range(int(args.rm_blocks)):
#     #         print(sort_list[i][1]) #.remove(drop_blocks)
#     #         drop_blocks.append(sort_list[i][1])
#     # lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
#     # print(f'=> latency reduction: {lat_reduction:.2f}%')
#     # recoverability,acc,_,_ = metric(metric_loader, pruned_model, origin_model, trained=False)
#     # print(acc)
#     # print(drop_blocks)
#     pruned_lat = 0
#     return pruned_model, drop_blocks, pruned_lat


def insert_one_block_adaptors_for_mobilenet(
    origin_model, prune_model, rm_block, params, args
):
    origin_named_modules = dict(origin_model.named_modules())
    pruned_named_modules = dict(prune_model.model.named_modules())

    print("-" * 50)
    print("=> {}".format(rm_block))
    has_rm_count = 0
    rm_channel = origin_named_modules[rm_block].out_channels
    key_items = rm_block.split(".")
    block_id = int(key_items[1])

    pre_block_id = block_id - has_rm_count - 1
    while pre_block_id > 0:
        pruned_module = pruned_named_modules[f"features.{pre_block_id}"]
        if rm_channel != pruned_module.out_channels:
            break
        last_conv_key = "features.{}.conv.2".format(pre_block_id)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        params.append({"params": conv.parameters()})
        pre_block_id -= 1
        # break

    after_block_id = block_id - has_rm_count
    while after_block_id < 18:
        pruned_module = pruned_named_modules[f"features.{after_block_id}"]
        after_conv_key = "features.{}.conv.0.0".format(after_block_id)
        conv = prune_model.add_preconv_for_conv(after_conv_key)
        params.append({"params": conv.parameters()})
        if rm_channel != pruned_module.out_channels:
            break
        after_block_id += 1
        # break

    has_rm_count += 1


def insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args):
    pruned_named_modules = dict(prune_model.model.named_modules())
    if "layer1.0.conv2" in pruned_named_modules:
        last_conv_in_block = "conv2"
    elif "layer1.0.conv3" in pruned_named_modules:
        last_conv_in_block = "conv3"
    else:
        raise ValueError("This is not a ResNet.")

    print("-" * 50)
    print("=> {}".format(rm_block))
    layer = ""
    block = ""
    if len(rm_block.split(".")) == 3:
        layer, block, _ = rm_block.split(".")
    else:
        layer, block = rm_block.split(".")
    rm_block_id = int(block)
    assert rm_block_id >= 1

    downsample = "{}.0.downsample.0".format(layer)
    if downsample in pruned_named_modules:
        conv = prune_model.add_afterconv_for_conv(downsample)
        if conv is not None:
            params.append({"params": conv.parameters()})

    for origin_block_num in range(rm_block_id):
        last_conv_key = "{}.{}.{}".format(layer, origin_block_num, last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({"params": conv.parameters()})

    for origin_block_num in range(rm_block_id + 1, 100):
        pruned_output_key = "{}.{}.conv1".format(layer, origin_block_num - 1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({"params": conv.parameters()})

    # next stage's conv1
    next_layer_conv1 = "layer{}.0.conv1".format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({"params": conv.parameters()})

    # next stage's downsample
    next_layer_downsample = "layer{}.0.downsample.0".format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({"params": conv.parameters()})


def insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args):
    rm_blocks_for_prune = []
    rm_blocks.sort()
    rm_count = [0, 0, 0, 0]
    layer = ""
    i = ""
    for block in rm_blocks:
        if len(block.split(".")) == 3:
            layer, i, _ = block.split(".")
        else:
            layer, i = block.split(".")
        l_id = int(layer[-1])
        b_id = int(i)
        prune_b_id = b_id - rm_count[l_id - 1]
        rm_count[l_id - 1] += 1
        rm_block_prune = f"{layer}.{prune_b_id}"
        rm_blocks_for_prune.append(rm_block_prune)
    
    # Add layer-wise learning rates for adaptors
    for rm_block in rm_blocks_for_prune:
        layer_num = int(rm_block.split('.')[0][-1])
        # Higher LR for layers closer to pruned block
        lr_multiplier = 2.0 if layer_num <= 2 else 1.0
        insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args, lr_multiplier)

def insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args, lr_multiplier=1.0):
    pruned_named_modules = dict(prune_model.model.named_modules())
    if "layer1.0.conv2" in pruned_named_modules:
        last_conv_in_block = "conv2"
    elif "layer1.0.conv3" in pruned_named_modules:
        last_conv_in_block = "conv3"
    else:
        raise ValueError("This is not a ResNet.")

    print("-" * 50)
    print("=> {} (LR multiplier: {:.1f})".format(rm_block, lr_multiplier))
    layer = ""
    block = ""
    if len(rm_block.split(".")) == 3:
        layer, block, _ = rm_block.split(".")
    else:
        layer, block = rm_block.split(".")
    rm_block_id = int(block)
    assert rm_block_id >= 1

    downsample = "{}.0.downsample.0".format(layer)
    if downsample in pruned_named_modules:
        conv = prune_model.add_afterconv_for_conv(downsample)
        if conv is not None:
            params.append({"params": conv.parameters(), "lr": args.lr * lr_multiplier})

    for origin_block_num in range(rm_block_id):
        last_conv_key = "{}.{}.{}".format(layer, origin_block_num, last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({"params": conv.parameters(), "lr": args.lr * lr_multiplier})

    for origin_block_num in range(rm_block_id + 1, 100):
        pruned_output_key = "{}.{}.conv1".format(layer, origin_block_num - 1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({"params": conv.parameters(), "lr": args.lr * lr_multiplier})

    # next stage's conv1
    next_layer_conv1 = "layer{}.0.conv1".format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({"params": conv.parameters(), "lr": args.lr * lr_multiplier})

    # next stage's downsample
    next_layer_downsample = "layer{}.0.downsample.0".format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({"params": conv.parameters(), "lr": args.lr * lr_multiplier})


def Practise_recover(train_loader, origin_model, prune_model, rm_blocks, args):
    params = []

    if "mobilenet" in args.model:
        assert len(rm_blocks) == 1
        insert_one_block_adaptors_for_mobilenet(
            origin_model, prune_model, rm_blocks[0], params, args
        )
    else:
        insert_all_adaptors_for_resnet(
            origin_model, prune_model, rm_blocks, params, args
        )

    # Enhanced initialization: Initialize adaptors with small random perturbations
    for param_dict in params:
        for param in param_dict['params']:
            if len(param.shape) == 4:  # Conv layer
                # Small perturbation around identity
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                param.data *= 0.1  # Scale down to stay close to identity
                
                # Add identity component
                if param.shape[0] == param.shape[1]:
                    eye = torch.eye(param.shape[0]).view(param.shape[0], param.shape[1], 1, 1)
                    param.data += eye.to(param.device)

    if args.opt == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.opt == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "AdamW":
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("{} not found".format(args.opt))

    # Add cosine annealing scheduler with warmup
    warmup_epochs = int(0.1 * args.epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch - warmup_epochs, eta_min=1e-6
    )

    recover_time = time.time()
    
    # Phase 1: Train adaptors only (first 40% of epochs)
    phase1_epochs = int(0.4 * args.epoch)
    print(f"Phase 1: Training adaptors only for {phase1_epochs} epochs")
    train_progressive(train_loader, optimizer, prune_model, origin_model, args, 
                     scheduler, warmup_epochs, phase1_epochs, phase=1, rm_blocks=rm_blocks)
    
    # Phase 2: Progressive unfreezing (remaining 60% of epochs)
    print(f"Phase 2: Progressive unfreezing with BN adaptation")
    unfreeze_nearby_bn_layers(prune_model, rm_blocks, params, optimizer)
    train_progressive(train_loader, optimizer, prune_model, origin_model, args, 
                     scheduler, warmup_epochs, args.epoch - phase1_epochs, 
                     phase=2, rm_blocks=rm_blocks, start_epoch=phase1_epochs)
    
    print(
        "compute recoverability {} takes {}s".format(
            rm_blocks, time.time() - recover_time
        )
    )


def unfreeze_nearby_bn_layers(prune_model, rm_blocks, params, optimizer):
    """Unfreeze BatchNorm layers near removed blocks for better adaptation"""
    model = prune_model.model if hasattr(prune_model, 'model') else prune_model
    
    # Get layers to unfreeze
    layers_to_unfreeze = set()
    for rm_block in rm_blocks:
        parts = rm_block.split('.')
        if len(parts) >= 2:
            layer_name = parts[0]  # e.g., 'layer1'
            layers_to_unfreeze.add(layer_name)
    
    # Unfreeze BN parameters in those layers
    bn_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for layer in layers_to_unfreeze:
                if name.startswith(layer):
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
                    bn_params.extend([module.weight, module.bias])
                    print(f"Unfroze BN layer: {name}")
                    break
    
    # Add BN parameters to optimizer
    if bn_params:
        optimizer.add_param_group({'params': bn_params, 'lr': optimizer.param_groups[0]['lr'] * 0.1})
        print(f"Added {len(bn_params)} BN parameters to optimizer")


def train_progressive(train_loader, optimizer, model, origin_model, args, 
                     scheduler=None, warmup_epochs=0, max_epochs=None, 
                     phase=1, rm_blocks=None, start_epoch=0):
    """Progressive training with multi-scale feature matching"""
    end = time.time()
    criterion = torch.nn.MSELoss(reduction="mean")
    
    # Multi-scale feature extraction hooks
    teacher_features_multi = {}
    student_features_multi = {}
    
    def get_activation(name, features_dict):
        def hook(module, input, output):
            features_dict[name] = output
        return hook
    
    # Register hooks for multi-scale features
    origin_model.cuda().eval()
    model.cuda().eval()
    
    # Hook intermediate layers for multi-scale matching
    if phase == 2:  # Only in phase 2
        for name, module in origin_model.named_modules():
            if 'layer' in name and len(name.split('.')) == 1:  # layer1, layer2, etc.
                module.register_forward_hook(get_activation(name, teacher_features_multi))
        
        for name, module in model.model.named_modules() if hasattr(model, 'model') else model.named_modules():
            if 'layer' in name and len(name.split('.')) == 1:
                module.register_forward_hook(get_activation(name, student_features_multi))
    
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"
    
    accumulation_steps = 2
    torch.cuda.empty_cache()
    iter_nums = start_epoch
    max_iters = max_epochs if max_epochs else args.epoch
    finish = False
    
    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > start_epoch + max_iters:
                finish = True
                break
            
            data_time.update(time.time() - end)
            
            if isinstance(data, torch.Tensor):
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6).cuda()
            else:
                data = safe_to_device(data)
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            
            with torch.no_grad():
                t_output, t_features = origin_model(data)
            
            if not assert_finite("t_features", t_features):
                print("Skipping batch due to non-finite teacher features")
                continue
            
            optimizer.zero_grad()
            output, s_features = model(data)
            
            if not assert_finite("s_features", s_features):
                print("Skipping batch due to non-finite student features")
                continue
            
            # Main feature loss
            loss = criterion(s_features, t_features)
            
            # Multi-scale feature matching in Phase 2
            if phase == 2 and teacher_features_multi and student_features_multi:
                multi_scale_loss = 0.0
                scale_count = 0
                for layer_name in teacher_features_multi:
                    if layer_name in student_features_multi:
                        t_feat = teacher_features_multi[layer_name]
                        s_feat = student_features_multi[layer_name]
                        
                        # Spatial pooling to match dimensions if needed
                        if t_feat.shape != s_feat.shape:
                            pool_size = t_feat.shape[2] // s_feat.shape[2] if t_feat.shape[2] > s_feat.shape[2] else 1
                            if pool_size > 1:
                                t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
                        
                        multi_scale_loss += criterion(s_feat, t_feat)
                        scale_count += 1
                
                if scale_count > 0:
                    multi_scale_loss /= scale_count
                    # Weighted combination: more emphasis on final features
                    loss = 0.7 * loss + 0.3 * multi_scale_loss
            
            if not assert_finite("loss", loss):
                print("Skipping batch due to non-finite loss")
                continue
            
            loss = loss / accumulation_steps
            losses.update(loss.data.item() * accumulation_steps, data.size(0))
            
            try:
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Learning rate scheduling
                    if warmup_epochs > 0 and iter_nums <= warmup_epochs + start_epoch:
                        lr = args.lr * ((iter_nums - start_epoch) / warmup_epochs)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    elif scheduler is not None and iter_nums > warmup_epochs + start_epoch:
                        scheduler.step()
            except Exception as e:
                print("Backward/step failed:", e)
                torch.cuda.empty_cache()
                continue
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if iter_nums % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    "Phase {5} Train: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    "LR {lr:.6f}".format(
                        iter_nums - start_epoch,
                        max_iters,
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                        lr=current_lr,
                        phase=phase
                    )
                )


def train(train_loader, optimizer, model, origin_model, args, scheduler=None, warmup_epochs=0):
    """Wrapper for backward compatibility"""
    train_progressive(train_loader, optimizer, model, origin_model, args, 
                     scheduler, warmup_epochs, max_epochs=args.epoch, phase=1, rm_blocks=[])


# def Practise_recover(
#     train_loader, metric_loader, origin_model, prune_model, rm_blocks, args
# ):
#     params = []
#     insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args)
#     print("\n" + "=" * 50)
#     print("Starting Practise_recover")
#     print(f"Removed blocks: {rm_blocks}")
#     print("=" * 50 + "\n")

#     # Initialize BlockReinitWarp
#     # reinit_model = BlockReinitWarp(prune_model)
#     reinit_model = prune_model

#     # print(f"Reinitializing blocks near {rm_blocks}")
#     # if isinstance(rm_blocks, str):
#     #     rm_blocks = [rm_blocks]
#     # elif not isinstance(rm_blocks, list):
#     #     rm_blocks = list(rm_blocks)

#     # print(f"Reinitializing blocks near: {rm_blocks}")
#     # reinit_model.reinitialize_nearby_blocks(rm_blocks, distance=1)

#     # reinit_model.freeze_non_trainable_blocks()

#     # params = reinit_model.get_trainable_parameters()
#     # if not params:
#     #     print("\nDEBUG INFO:")
#     #     print(f"rm_blocks type: {type(rm_blocks)}")
#     #     print(f"rm_blocks content: {rm_blocks}")
#     #     raise ValueError("No trainable parameters found! Check if blocks were properly reinitialized.")

#     if args.opt == "SGD":
#         optimizer = torch.optim.SGD(
#             params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
#         )
#     elif args.opt == "Adam":
#         optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#     elif args.opt == "AdamW":
#         optimizer = torch.optim.AdamW(
#             params, lr=args.lr, weight_decay=args.weight_decay
#         )
#     else:
#         raise ValueError("{} not found".format(args.opt))

#     recover_time = time.time()

#     # Initial training
#     # train_clkd(train_loader, metric_loader, optimizer, reinit_model, origin_model, args)
#     train_clkd(train_loader, metric_loader, optimizer, reinit_model, origin_model, args)
#     # train(train_loader, metric_loader, optimizer, reinit_model, origin_model, args)
#     # Check for problematic classes
#     # _, acc, _ = metric(metric_loader, model, origin_model)
#     #     print(f"Current Accuracy: {acc:.4f}")
#     #     model.train()  # Switch back to train mode after evaluation


def nmse_loss(p, z):
    p_norm = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mean((p_norm - z_norm) ** 2)


def class_correlation_matrix(Z):
    """
    Z: [B, C] feature matrix (batch x channels)
    returns: [C x C] class correlation matrix
    """
    Z = Z.view(Z.size(0), -1).contiguous()  # Flatten and ensure 2D [B, C]
    Z_mean = Z.mean(dim=0, keepdim=True)  # [1, C]
    Z_centered = Z - Z_mean  # [B, C]
    return Z_centered.T @ Z_centered / (Z.size(0) - 1)


def train_clkd(
    train_loader,
    metric_loader,
    optimizer,
    model,
    origin_model,
    args,
    problematic_classes=None,
):
    end = time.time()
    ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adaptive loss weights with warmup
    warmup_epochs = int(0.1 * args.epoch)
    
    # Temperature for knowledge distillation
    temperature = 4.0
    
    def get_loss_weights(current_iter):
        if current_iter < warmup_epochs:
            alpha = current_iter / warmup_epochs
            lambda_ce = 0.2 + 0.1 * alpha
            lambda_kd = 0.3 + 0.2 * alpha
            mu_nmse = 0.3 + 0.2 * alpha
            nu_cc = 0.05 + 0.15 * alpha
        else:
            lambda_ce = 0.2
            lambda_kd = 0.5
            mu_nmse = 0.4
            nu_cc = 0.2
        return lambda_ce, lambda_kd, mu_nmse, nu_cc

    # Extract features from pre-GAP layer
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"

    model.cuda().train()
    origin_model.cuda().eval()

    iter_nums = 0
    finish = False
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    while not finish:
        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > args.epoch:
                finish = True
                break

            # sanitize inputs
            if isinstance(data, torch.Tensor):
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6).cuda()
            else:
                data = safe_to_device(data)

            target = target.cuda()
            data_time.update(time.time() - end)

            with torch.no_grad():
                _, t_features = origin_model(data)

            # check teacher features
            if not assert_finite("t_features", t_features):
                print(f"[batch {iter_nums}] skipping: teacher features non-finite")
                continue

            _, s_features = model(data)
            s_logits, _ = model(data)

            # Basic finiteness checks
            if not assert_finite("s_features", s_features):
                print(f"[batch {iter_nums}] skipping: student features non-finite")
                continue
            if not assert_finite("s_logits", s_logits):
                print(f"[batch {iter_nums}] skipping: student logits non-finite")
                continue

            ce_loss = ce_criterion(s_logits, target)
            if not assert_finite("ce_loss", ce_loss):
                print(f"[batch {iter_nums}] skipping: ce_loss non-finite")
                continue

            # Soft target KD loss with temperature scaling
            t_soft = F.softmax(t_logits / temperature, dim=1)
            s_soft = F.log_softmax(s_logits / temperature, dim=1)
            kd_soft_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)
            
            if not assert_finite("kd_soft_loss", kd_soft_loss):
                print(f"[batch {iter_nums}] skipping: kd_soft_loss non-finite")
                continue

            l_ins = nmse_loss(s_features, t_features)
            if problematic_classes is None:
                l_cla = nmse_loss(s_features.T, t_features.T)
                cc_s = class_correlation_matrix(s_features)
                cc_t = class_correlation_matrix(t_features)
                cc_loss = torch.mean((cc_s - cc_t) ** 2)
            else:
                l_cla, cc_loss = compute_focused_class_losses(
                    s_features, t_features, target, problematic_classes
                )

            # Get adaptive loss weights
            lambda_ce, lambda_kd, mu_nmse, nu_cc = get_loss_weights(iter_nums)
            
            kd_feature_loss = l_ins + l_cla
            total_loss = lambda_ce * ce_loss + lambda_kd * kd_soft_loss + mu_nmse * kd_feature_loss + nu_cc * cc_loss

            if not assert_finite("total_loss", total_loss):
                print(f"[batch {iter_nums}] skipping: total_loss non-finite")
                continue

            optimizer.zero_grad()
            try:
                # enable anomaly detection around backward to get op stack if needed
                with torch.autograd.set_detect_anomaly(True):
                    total_loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
                )
                optimizer.step()
            except RuntimeError as e:
                print(f"[batch {iter_nums}] backward failed: {e}")
                # optionally save offending batch for inspection
                try:
                    torch.save(
                        {"data": data.detach().cpu(), "target": target.detach().cpu()},
                        f"bad_batch_{iter_nums}.pt",
                    )
                    print(f"Saved bad batch bad_batch_{iter_nums}.pt")
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue

            losses.update(total_loss.item(), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if iter_nums % 50 == 0:
                print(
                    f"Train: [{iter_nums}/{args.epoch}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})"
                )


def compute_focused_class_losses(s_features, t_features, targets, focused_classes):
    """
    Compute class NMSE and correlation loss for selected classes only.
    """
    # Flatten features to [B, C]
    s_feat = s_features.view(s_features.size(0), -1).contiguous()
    t_feat = t_features.view(t_features.size(0), -1).contiguous()

    # Mask for focused classes
    mask = torch.tensor(
        [c in focused_classes for c in targets.cpu().tolist()], device=targets.device
    )
    if mask.sum() < 2:
        # Not enough samples in batch from focused classes, skip
        return 0.0, 0.0

    s_selected = s_feat[mask]
    t_selected = t_feat[mask]

    # Class-level NMSE (transpose)
    class_nmse = nmse_loss(s_selected.T, t_selected.T)

    # Class correlation loss
    cc_s = class_correlation_matrix(s_selected)
    cc_t = class_correlation_matrix(t_selected)
    cc_loss = torch.mean((cc_s - cc_t) ** 2)

    return class_nmse, cc_loss


def metric(metric_loader, model, origin_model, trained=False):
    criterion = torch.nn.MSELoss(reduction="mean")

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    origin_model.get_feat = "pre_GAP"
    model.cuda()
    model.eval()
    model.get_feat = "pre_GAP"

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    origin_accuracies = AverageMeter()

    # Initialize per-category accuracy tracking
    num_classes = 1000  # Assuming ImageNet classes
    correct_per_class = torch.zeros(num_classes).cuda()
    total_per_class = torch.zeros(num_classes).cuda()
    origin_correct_per_class = torch.zeros(num_classes).cuda()
    # Initialize per-class MSE loss tracking
    mse_loss_per_class = torch.zeros(num_classes).cuda()

    end = time.time()
    for i, (data, target) in enumerate(metric_loader):
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)
            loss = criterion(s_features, t_features)

            # Calculate overall accuracy
            acc = accuracy(s_output, target, topk=(1,))[0]
            origin_acc = accuracy(t_output, target, topk=(1,))[0]

        losses.update(loss.data.item(), data.size(0))
        accuracies.update(acc.item(), data.size(0))
        origin_accuracies.update(origin_acc.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print(
                "Metric: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                "Accuracy {accuracies.val:.2f} ({accuracies.avg:.2f})".format(
                    i,
                    len(metric_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    losses=losses,
                    accuracies=accuracies,
                )
            )

    print(" * Metric Loss {loss.avg:.4f}".format(loss=losses))

    problematic_classes = []
    print(
        f"Overall Accuracy - Pruned: {accuracies.avg:.2f}%, Original: {origin_accuracies.avg:.2f}%"
    )

    return losses.avg, accuracies.avg, origin_accuracies.avg, problematic_classes


def train_focused(
    train_loader, metric_loader, optimizer, model, origin_model, args, target_classes
):
    """
    Train the model focusing on specific classes with large accuracy differences
    """
    criterion = torch.nn.MSELoss(reduction="mean")
    cls_criterion = torch.nn.CrossEntropyLoss()

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.train()

    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"

    torch.cuda.empty_cache()
    iter_nums = 0
    finish = False

    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > args.epoch:
                finish = True
                break

            # Filter data for target classes
            mask = torch.tensor(
                [t in target_classes for t in target], device=data.device
            )
            if not mask.any():
                continue

            data = data[mask].cuda()
            target = target[mask].cuda()

            with torch.no_grad():
                t_logits, t_features = origin_model(data)
                t_probs = nn.functional.softmax(t_logits / 1.0, dim=1)

            optimizer.zero_grad()
            s_logits, s_features = model(data)

            # Feature matching loss
            feat_loss = criterion(s_features, t_features)

            # Classification loss with higher weight for target classes
            # cls_loss = cls_criterion(s_logits, target)

            # Combined loss with higher weight for classification
            loss = feat_loss

            losses.update(loss.item(), data.size(0))

            loss.backward()
            optimizer.step()

            if iter_nums % 50 == 0:
                print(
                    "Focused Train: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})".format(
                        iter_nums,
                        args.epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                    )
                )

            if iter_nums % 400 == 0:
                _, acc, _, _ = metric(metric_loader, model, origin_model)
                print(f"Current Accuracy: {acc:.4f}")
                model.train()

    return model


def log_model_parameters(model, model_name="Model", log_file="model_parameters.log"):
    # Check for existing files and increment the file name if necessary
    base_name, ext = os.path.splitext(log_file)
    counter = 1
    while os.path.exists(log_file):
        log_file = f"{base_name}_{counter}{ext}"
        counter += 1

    # Set PyTorch print options to display all elements
    torch.set_printoptions(
        edgeitems=None, linewidth=1000, sci_mode=False, threshold=float("inf")
    )

    with open(log_file, "a") as f:  # Open the log file in append mode
        f.write(f"Parameters of {model_name}:\n")
        for name, param in model.named_parameters():
            f.write(
                f"  {name} -> Shape: {param.shape}, Requires Grad: {param.requires_grad}\n"
            )
            f.write(f"{param}\n")  # Logs the full tensor data
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total Parameters in {model_name}: {total_params}\n\n")

    # Optionally, reset print options to default after logging
    torch.set_printoptions(edgeitems=3, linewidth=80, sci_mode=None, threshold=1000)

    print(f"Parameters logged in {log_file}")


def example_batchnorm_gamma_analysis(model):
    """
    Example function showing how to use the BatchNorm gamma analysis functions.

    Args:
        model: PyTorch ResNet model
    """
    print("Example: Analyzing BatchNorm Gamma Values")
    print("=" * 50)

    # Method 1: Get raw gamma values
    print("\n1. Getting raw gamma values...")
    gamma_values = get_batchnorm_gamma_values(model)
    print(f"Found {len(gamma_values)} BatchNorm layers")

    # Method 2: Print summary
    print("\n2. Printing summary...")
    print_batchnorm_gamma_summary(model)

    # Method 3: Save to file
    print("\n3. Saving to file...")
    filename = save_batchnorm_gamma_to_file(model)

    # Method 4: Create visualizations
    print("\n4. Creating visualizations...")
    analysis_results = analyze_batchnorm_gamma_distribution(model, save_plot=True)

    # Method 5: Access specific layer gamma values
    print("\n5. Accessing specific layer gamma values...")
    if gamma_values:
        # Get the first BatchNorm layer as an example
        first_bn_name = list(gamma_values.keys())[0]
        first_bn_data = gamma_values[first_bn_name]
        print(f"First BatchNorm layer: {first_bn_name}")
        print(f"  Gamma values shape: {first_bn_data['gamma_values'].shape}")
        print(f"  Mean gamma: {first_bn_data['mean_gamma']:.4f}")
        print(f"  First 5 gamma values: {first_bn_data['gamma_values'][:5]}")

    return gamma_values, analysis_results


def measure_taylor_saliency_per_block(
    model, data_loader, num_batches=10, device="cuda"
):
    """
    Compute Taylor expansion-based saliency score per residual block.

    The first-order Taylor approximation of loss change when a block's weights
    are zeroed is: sum(|w_i * grad_i|) over all parameters in the block.

    This measures how much the loss would change if the block were removed.
    Higher score => more important block (larger loss change if removed).

    Args:
        model: PyTorch model (e.g., ResNet)
        data_loader: DataLoader yielding (images, labels)
        num_batches: Number of batches to use for estimation
        device: 'cuda' or 'cpu'

    Returns:
        OrderedDict: {block_name: taylor_saliency_score}
    """
    model.eval()  # keep BN/Dropout deterministic; gradients still computed
    if device:
        model.to(device)

    # Collect residual block names in order
    block_names = []
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            blk = ".".join(name.split(".")[:2])
            if blk not in block_names:
                block_names.append(blk)

    saliency_sums = collections.OrderedDict((bn, 0.0) for bn in block_names)
    sample_count = 0

    batch_idx = 0
    for images, targets in data_loader:
        if batch_idx >= num_batches:
            break
        batch_idx += 1

        if device:
            images = images.to(device)
            targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Use cross-entropy loss for classification
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        # Backward pass to compute gradients
        loss.backward()

        # Compute Taylor saliency: sum(|w_i * grad_i|) per block
        for pname, p in model.named_parameters():
            if p.grad is None:
                continue

            # Identify which block this parameter belongs to
            parts = pname.split(".")
            blk = None
            if len(parts) >= 2 and parts[0].startswith("layer"):
                blk = parts[0] + "." + parts[1]

            if blk is None or blk not in saliency_sums:
                continue

            # Taylor saliency: |weight * gradient|
            saliency = torch.abs(p.data * p.grad.detach()).sum().item()
            saliency_sums[blk] += saliency

        sample_count += images.size(0)

    # Average per sample processed
    if sample_count > 0:
        for k in saliency_sums:
            saliency_sums[k] = saliency_sums[k] / sample_count

    return saliency_sums


# if __name__ == '__main__':
#     main()
