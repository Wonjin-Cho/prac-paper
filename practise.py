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
# import matplotlib.pyplot as plt  # Commented out due to XML parser import error
# import seaborn as sns  # Commented out due to XML parser import error
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
from amfr_cr import AMFRCRLoss, extract_multi_scale_features
from dgkd import DGKDLoss, extract_features_for_dgkd


class ModelEMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.9999, use_cuda=False):
        self.module = copy.deepcopy(model)
        if use_cuda and torch.cuda.is_available():
            self.module.cuda()
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(),
                                              model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data,
                                                     alpha=1 - self.decay)

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
# import matplotlib.pyplot as plt  # Commented out due to XML parser import error
# import seaborn as sns  # Commented out due to XML parser import error
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
from amfr_cr import AMFRCRLoss, extract_multi_scale_features
from dgkd import DGKDLoss, extract_features_for_dgkd


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
            print(
                f"[NaN/Inf] {name} has non-finite values (min={mn}, max={mx})")
            return False
    except Exception:
        print(f"[assert_finite] could not check {name} (type={type(tensor)})")
        return False
    return True


def safe_to_device(x, device):
    """Convert numpy or tensor to torch tensor on device."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


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
        [data["gamma_values"] for data in gamma_values.values()])
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


def save_batchnorm_gamma_to_file(model,
                                 filename="batchnorm_gamma_values.pickle"):
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
        # Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
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
        max_sim_idx = np.unravel_index(np.argmax(similarity_np),
                                       similarity_np.shape)
        print(
            f"Most similar channels: {max_sim_idx} with similarity {similarity_np[max_sim_idx]:.4f}")

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


def get_block_features(args, model, block_name, data_loader, num_batches=10):
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
            if args.cuda:
                data = data.cuda()
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            _ = model(data)

            # Store features
            if input_hook.outputs is not None:
                # Flatten spatial dimensions and average across batch
                input_feat = input_hook.outputs.view(
                    input_hook.outputs.size(0), -1)
                input_features.append(input_feat.mean(dim=0).cpu())
            if output_hook.outputs is not None:
                # Flatten spatial dimensions and average across batch
                output_feat = output_hook.outputs.view(
                    output_hook.outputs.size(0), -1)
                output_features.append(output_feat.mean(dim=0).cpu())

    # Average features across batches
    input_features = torch.stack(input_features).mean(dim=0)
    output_features = torch.stack(output_features).mean(dim=0)

    return input_features, output_features


def compute_block_similarities(args,
                               model,
                               target_block,
                               data_loader,
                               num_batches=10):
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
    target_input, target_output = get_block_features(args, model, target_block,
                                                     data_loader, num_batches)

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
                args, model, block_name, data_loader, num_batches)

            # Ensure features have the same dimension
            min_dim = min(target_input.size(0), block_input.size(0))
            target_input_trimmed = target_input[:min_dim]
            block_input_trimmed = block_input[:min_dim]

            min_dim = min(target_output.size(0), block_output.size(0))
            target_output_trimmed = target_output[:min_dim]
            block_output_trimmed = block_output[:min_dim]

            # Compute cosine similarities
            input_sim = F.cosine_similarity(target_input_trimmed.unsqueeze(0),
                                            block_input_trimmed.unsqueeze(0))
            output_sim = F.cosine_similarity(
                target_output_trimmed.unsqueeze(0),
                block_output_trimmed.unsqueeze(0))

            similarities["input"][block_name] = input_sim.item()
            similarities["output"][block_name] = output_sim.item()

    return similarities


def compute_block_io_similarity(args,
                                model,
                                block_name,
                                data_loader,
                                num_batches=10):
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
        args, model, block_name, data_loader, num_batches)

    # Ensure features have the same dimension
    min_dim = min(input_features.size(0), output_features.size(0))
    input_features = input_features[:min_dim]
    output_features = output_features[:min_dim]

    # Compute cosine similarity
    similarity = F.cosine_similarity(input_features.unsqueeze(0),
                                     output_features.unsqueeze(0))

    return similarity.item()


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


def Practise_one_block(rm_block,
                       origin_model,
                       origin_lat,
                       train_loader,
                       metric_loader,
                       args,
                       drop_blocks=0):
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()

    # Check if rm_block is a list with multiple blocks
    if isinstance(rm_block, list) and len(rm_block) > 1:
        print(f"\n{'='*60}")
        print(f"PROGRESSIVE ITERATIVE PRUNING: {len(rm_block)} blocks")
        print(f"Blocks to prune: {rm_block}")
        print(f"{'='*60}\n")

        # Use progressive iterative pruning for better performance
        pruned_model, recoverability, lat_reduction, score = progressive_iterative_pruning(
            rm_block, origin_model, origin_lat, train_loader, metric_loader,
            args)
        return pruned_model, (recoverability, lat_reduction, score)

    # Single block pruning (original logic)
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

    pruned_model_adaptor = AdaptorWarp(pruned_model)

    start_time = time.time()
    Practise_recover(train_loader, origin_model, pruned_model_adaptor,
                     rm_block, args)
    print("Total time: {:.3f}s".format(time.time() - start_time))

    print("Metric w/ Recovering:")
    recoverability = metric(args, metric_loader, pruned_model_adaptor,
                            origin_model)
    pruned_model_adaptor.remove_all_preconv()
    pruned_model_adaptor.remove_all_afterconv()

    score = 0
    device = "cuda"
    return pruned_model, (recoverability, lat_reduction, score)


def progressive_iterative_pruning(rm_blocks, origin_model, origin_lat,
                                  train_loader, metric_loader, args):
    """
    Progressive iterative pruning: Remove blocks one-by-one with intermediate training.
    This helps the model adapt gradually to capacity reduction.

    Args:
        rm_blocks: List of blocks to remove (e.g., ['layer1.1', 'layer1.2', 'layer3.3'])
        origin_model: Teacher model
        origin_lat: Original latency
        train_loader: Training data loader
        metric_loader: Validation data loader
        args: Arguments

    Returns:
        Final pruned model with all blocks removed
    """
    # Sort blocks by layer to prune from early to late layers
    sorted_blocks = sorted(rm_blocks,
                           key=lambda x:
                           (int(x.split('.')[0][-1]), int(x.split('.')[1])))
    print(f"Pruning order: {sorted_blocks}")

    # Store epochs for progressive training
    original_epochs = args.epoch
    # Allocate more epochs for synthetic data: 40% per block (1000 epochs per block)
    per_block_epochs = max(1000, int(original_epochs * 0.4))

    current_model = origin_model
    cumulative_blocks = []

    for i, block in enumerate(sorted_blocks):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}/{len(sorted_blocks)}: Pruning block '{block}'")
        print(f"Previously pruned: {cumulative_blocks}")
        print(f"{'='*60}\n")

        cumulative_blocks.append(block)

        # Build student with cumulative blocks removed
        pruned_model, _, pruned_lat = build_student(
            args.model,
            cumulative_blocks,
            args.num_classes,
            state_dict_path=args.state_dict_path,
            teacher=args.teacher,
            cuda=args.cuda,
        )

        lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
        print(f"=> Cumulative latency reduction: {lat_reduction:.2f}%")

        # Wrap with adaptors
        pruned_model_adaptor = AdaptorWarp(pruned_model)

        # Progressive epoch allocation: more epochs for later iterations
        if i == len(sorted_blocks) - 1:
            # Final iteration gets 2x epochs for thorough fine-tuning
            current_epochs = int(per_block_epochs * 2.0)
        elif i == 0:
            # First iteration: moderate epochs to avoid overfitting to synthetic data
            current_epochs = per_block_epochs
        else:
            # Middle iterations: 1.2x epochs
            current_epochs = int(per_block_epochs * 1.2)

        # Temporarily update args.epoch
        args.epoch = current_epochs
        print(f"Training for {current_epochs} epochs")

        # Recover with progressive training
        start_time = time.time()
        Practise_recover(train_loader, current_model, pruned_model_adaptor,
                         cumulative_blocks, args)
        print(f"Recovery time: {time.time() - start_time:.3f}s")

        # Evaluate intermediate performance
        print(f"\nEvaluating after pruning block '{block}':")
        recoverability, acc, origin_acc, _ = metric(args, metric_loader,
                                                    pruned_model_adaptor,
                                                    origin_model)
        print(
            f"Recoverability: {recoverability:.4f}, Acc: {acc:.2f}%, Teacher Acc: {origin_acc:.2f}%"
        )

        # Remove adaptors and absorb them into the model
        pruned_model_adaptor.remove_all_preconv(absorb=True)
        pruned_model_adaptor.remove_all_afterconv(absorb=True)

        # Update current model for next iteration (use pruned model as new teacher)
        current_model = pruned_model_adaptor.model

        # Clear cache
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()

    # Restore original epochs
    args.epoch = original_epochs

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION after all blocks pruned")
    print(f"{'='*60}\n")

    final_recoverability, final_acc, origin_acc, _ = metric(
        args, metric_loader, current_model, origin_model)

    # Calculate final latency reduction
    _, _, final_lat = build_student(
        args.model,
        cumulative_blocks,
        args.num_classes,
        state_dict_path=args.state_dict_path,
        teacher=args.teacher,
        cuda=args.cuda,
    )
    final_lat_reduction = (origin_lat - final_lat) / origin_lat * 100

    print(f"\nProgressive Iterative Pruning Complete!")
    print(f"Final Accuracy: {final_acc:.2f}% (Teacher: {origin_acc:.2f}%)")
    print(f"Final Latency Reduction: {final_lat_reduction:.2f}%")
    print(f"Total blocks pruned: {len(cumulative_blocks)}")

    score = final_acc  # Use accuracy as score

    return current_model, final_recoverability, final_lat_reduction, score


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
        _, results = Practise_one_block(rm_block, origin_model, origin_lat,
                                        train_loader, metric_loader, args)
        recoverabilities[rm_block] = results

    print("-" * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score = recoverabilities[block]
        print(
            f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
        sort_list.append([score, block])
    print("-" * 50)
    print("=> sorted")
    sort_list.sort()
    for score, block in sort_list:
        print(f"{block} -> {score:.4f}")
    print("-" * 50)
    print(
        f"=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})"
    )
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


def insert_one_block_adaptors_for_mobilenet(origin_model, prune_model,
                                            rm_block, params, args):
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
        last_conv_key = "{}.{}.{}".format(layer, origin_block_num,
                                          last_conv_in_block)
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


def insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks,
                                   params, args):
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
        insert_one_block_adaptors_for_resnet(prune_model, rm_block, params,
                                             args, lr_multiplier)


def insert_one_block_adaptors_for_resnet(prune_model,
                                         rm_block,
                                         params,
                                         args,
                                         lr_multiplier=1.0):
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
            params.append({
                "params": conv.parameters(),
                "lr": args.lr * lr_multiplier
            })

    for origin_block_num in range(rm_block_id):
        last_conv_key = "{}.{}.{}".format(layer, origin_block_num,
                                          last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({
                "params": conv.parameters(),
                "lr": args.lr * lr_multiplier
            })

    for origin_block_num in range(rm_block_id + 1, 100):
        pruned_output_key = "{}.{}.conv1".format(layer, origin_block_num - 1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({
                "params": conv.parameters(),
                "lr": args.lr * lr_multiplier
            })

    # next stage's conv1
    next_layer_conv1 = "layer{}.0.conv1".format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({
                "params": conv.parameters(),
                "lr": args.lr * lr_multiplier
            })

    # next stage's downsample
    next_layer_downsample = "layer{}.0.downsample.0".format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({
                "params": conv.parameters(),
                "lr": args.lr * lr_multiplier
            })


def Practise_recover(train_loader, origin_model, prune_model, rm_blocks, args):
    params = []

    if "mobilenet" in args.model:
        assert len(rm_blocks) == 1
        insert_one_block_adaptors_for_mobilenet(origin_model, prune_model,
                                                rm_blocks[0], params, args)
    else:
        insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks,
                                       params, args)

    # Detect aggressive pruning (3+ blocks)
    num_blocks = len(rm_blocks) if rm_blocks and isinstance(rm_blocks,
                                                            list) else 1
    is_aggressive_pruning = num_blocks >= 3

    if is_aggressive_pruning:
        print(f"\n⚠️  Aggressive pruning detected ({num_blocks} blocks)")
        print("Using enhanced training strategy with:")
        print("  - Curriculum learning (easy→hard)")
        print("  - Adaptive learning rate scheduling")
        print("  - Stronger feature matching")
        print("  - Gradient accumulation\n")

    # Create EMA model for self-distillation
    ema_model = ModelEMA(prune_model, decay=0.9996, use_cuda=args.cuda)

    # Enhanced initialization: Initialize adaptors with small random perturbations
    for param_dict in params:
        for param in param_dict['params']:
            if len(param.shape) == 4:  # Conv layer
                # Small perturbation around identity
                nn.init.kaiming_normal_(param,
                                        mode='fan_out',
                                        nonlinearity='relu')
                param.data *= 0.1  # Scale down to stay close to identity

                # Add identity component
                if param.shape[0] == param.shape[1]:
                    eye = torch.eye(param.shape[0]).view(
                        param.shape[0], param.shape[1], 1, 1)
                    param.data += eye.to(param.device)

    # Define optimizer and scheduler
    optimizer_params = params if params else model.parameters()
    if args.opt == "SGD":
        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"{args.opt} not found")

    # Add cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01)

    # Define warmup epochs for aggressive pruning
    warmup_epoch = 5 if is_aggressive_pruning else 0

    # Detect aggressive pruning (3+ blocks)
    num_blocks = len(rm_blocks) if rm_blocks and isinstance(rm_blocks,
                                                            list) else 1
    # is_aggressive_pruning = num_blocks >= 3 # This line is now redundant

    # For synthetic data: Balanced phase allocation
    # Phase 1: Train adaptors only (first 60% of epochs for synthetic data)
    phase1_epochs = int(0.6 * args.epoch)  # Increased from 0.5
    print(
        f"Phase 1: Training adaptors only for {phase1_epochs} epochs (synthetic data mode)"
    )
    print(
        "Using noise-robust Huber loss and quality-weighted feature matching")
    train_progressive(train_loader,
                      optimizer,
                      prune_model,
                      origin_model,
                      args,
                      scheduler,
                      warmup_epoch,
                      phase1_epochs,
                      phase=1,
                      rm_blocks=rm_blocks)

    # Phase 2: Progressive unfreezing (remaining 40% of epochs)
    print(
        f"Phase 2: Progressive unfreezing with BN adaptation (synthetic data mode)"
    )
    unfreeze_nearby_bn_layers(prune_model, rm_blocks, params, optimizer)

    # Reset scheduler for phase 2 with lower learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * 0.5  # Half the learning rate for phase 2

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epoch -
                                                           phase1_epochs,
                                                           eta_min=1e-7)

    train_progressive(train_loader,
                      optimizer,
                      prune_model,
                      origin_model,
                      args,
                      scheduler,
                      warmup_epoch,
                      args.epoch - phase1_epochs,
                      phase=2,
                      rm_blocks=rm_blocks,
                      start_epoch=phase1_epochs)

    # compute recoverability # This was inside the original function, but now it's commented out.
    # The metric function is called later.
    # recover_time = time.time()
    # compute recoverability
    # print("compute recoverability {} takes {}s".format(
    #     rm_blocks,
    #     time.time() - recover_time))


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
        optimizer.add_param_group({
            'params': bn_params,
            'lr': optimizer.param_groups[0]['lr'] * 0.1
        })
        print(f"Added {len(bn_params)} BN parameters to optimizer")


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    if x.is_cuda:
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    index = index.to(y.device)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def feature_statistics_loss(student_feat, teacher_feat):
    """Match channel-wise mean and variance"""
    # Compute per-channel statistics
    s_mean = student_feat.mean(dim=[0, 2, 3])
    t_mean = teacher_feat.mean(dim=[0, 2, 3])

    s_var = student_feat.var(dim=[0, 2, 3])
    t_var = teacher_feat.var(dim=[0, 2, 3])

    mean_loss = F.mse_loss(s_mean, t_mean)
    var_loss = F.mse_loss(s_var, t_var)

    return mean_loss + var_loss


def spatial_attention_loss(student_feat, teacher_feat):
    """Compute attention-weighted feature matching loss"""
    # Compute spatial attention maps from teacher
    B, C, H, W = teacher_feat.shape
    teacher_attention = torch.mean(teacher_feat, dim=1,
                                   keepdim=True)  # [B, 1, H, W]
    teacher_attention = F.softmax(teacher_attention.view(B, -1),
                                  dim=1).view(B, 1, H, W)

    # Weight the feature difference by attention
    weighted_diff = teacher_attention * (student_feat - teacher_feat)**2
    return weighted_diff.mean()


def extract_features_from_model(model, x, layer_names, device):
    """
    Extract features from specified layers of the model.

    Args:
        model: The neural network model
        x: Input tensor
        layer_names: List of layer names to extract features from (e.g., ['layer1', 'layer2'])
        device: Device to run the model on

    Returns:
        features_dict: Dictionary mapping layer names to their output features
        logits: Final model output (logits)
    """
    features_dict = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    # Get the named modules once
    named_modules = dict(model.named_modules())

    # Register hooks for each layer
    # Handle both full layer names (e.g., 'layer1.0') and layer group names (e.g., 'layer1')
    for name in layer_names:
        if name in named_modules:
            # Direct match (e.g., 'layer1.0')
            module = named_modules[name]
            hooks.append(module.register_forward_hook(get_hook(name)))
        else:
            # Try to find the last block in the layer (e.g., 'layer1' -> get the Sequential layer itself)
            # First, check if it's a layer group (layer1, layer2, etc.)
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                # For ResNet, these are nn.Sequential containers
                # We want to hook the Sequential container itself to get output after all its blocks
                layer_module = getattr(model, name, None)
                if layer_module is not None:
                    hooks.append(layer_module.register_forward_hook(get_hook(name)))
                else:
                    # Try to access through model.model if it's wrapped
                    if hasattr(model, 'model'):
                        layer_module = getattr(model.model, name, None)
                        if layer_module is not None:
                            hooks.append(layer_module.register_forward_hook(get_hook(name)))
                        else:
                            print(f"Warning: Could not find module for layer name: {name}")
                    else:
                        print(f"Warning: Could not find module for layer name: {name}")
            else:
                # Try to find matching modules that start with this name
                matching_modules = [(k, v) for k, v in named_modules.items() if k.startswith(name + '.')]
                if matching_modules:
                    # Get the last block (highest numbered block)
                    last_module_name = max([k for k, v in matching_modules], key=lambda k: k.split('.')[-1] if k.split('.')[-1].isdigit() else -1)
                    module = named_modules[last_module_name]
                    hooks.append(module.register_forward_hook(get_hook(name)))
                else:
                    print(f"Warning: Could not find module for layer name: {name}")

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return features_dict, logits


def train_progressive(train_loader,
                      optimizer,
                      model,
                      origin_model,
                      args,
                      scheduler=None,
                      warmup_epochs=0,
                      max_epochs=None,
                      phase=1,
                      rm_blocks=None,
                      start_epoch=0):
    """Progressive training with multi-scale feature matching, adapted for noisy synthetic data"""
    end = time.time()

    # Determine device at the start
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Detect aggressive pruning (3+ blocks)
    num_blocks = len(rm_blocks) if rm_blocks and isinstance(rm_blocks,
                                                            list) else 1
    is_aggressive_pruning = num_blocks >= 3

    # Use Huber loss instead of MSE for robustness to outliers in synthetic data
    criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)

    # AMFR-CR Loss components
    # Define layer names and channels for AMFR-CR
    # For ResNet34: layer1 has 64 channels, layer2 has 128, layer3 has 256, layer4 has 512
    teacher_channels = [64, 128, 256]  # First 3 layers
    student_channels = [64, 128, 256]  # Same for student (matching layers)
    feature_dim = 256  # Use layer3 output dimension

    # Option 1: Use simplified loss for better stability
    use_simplified = True  # Set to False to use full AMFR-CR

    if use_simplified:
        from amfr_cr import SimplifiedFeatureLoss
        amfr_criterion = SimplifiedFeatureLoss(
            teacher_channels=teacher_channels[:3],
            student_channels=student_channels,
            feature_dim=512
        )
    else:
        amfr_criterion = AMFRCRLoss(
            teacher_channels=teacher_channels[:3],
            student_channels=student_channels,
            feature_dim=512,
            use_curriculum=False  # Disable curriculum for stability
        )

    # DGKD Loss
    # Define layer names and channels for DGKD
    # Assuming similar layer structure for DGKD as AMFRCR
    dgkd_criterion = DGKDLoss(
        teacher_channels=teacher_channels[:3],
        student_channels=student_channels,
        num_epochs=args.epochs
    )


    # Move criterion to device if using CUDA
    if args.cuda:
        amfr_criterion = amfr_criterion.cuda()
        dgkd_criterion = dgkd_criterion.cuda()

    # Curriculum learning: Start with easier examples, gradually increase difficulty
    if is_aggressive_pruning and phase == 1:
        print("🎓 Curriculum Learning: Starting with easier samples")
        curriculum_stage = 0  # 0=easy, 1=medium, 2=hard
        curriculum_switch_iter = max_epochs // 3  # Switch every 1/3 of training

    # Early stopping for synthetic data (prevent overfitting)
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 100  # Stop if no improvement for 100 iterations

    # Define layer names for feature extraction
    layer_names_teacher = ['layer1', 'layer2', 'layer3', 'layer4']
    layer_names_student = [
        'layer1', 'layer2', 'layer3'
    ]  # Adjusted for AMFRCR loss to match channels

    # Move models to device
    origin_model.to(device)
    model.to(device)

    # Set models to evaluation mode (for feature extraction)
    origin_model.eval()
    model.eval()

    accumulation_steps = 2
    if args.cuda:
        torch.cuda.empty_cache()
    iter_nums = start_epoch
    max_iters = max_epochs if max_epochs else args.epoch
    finish = False

    # Calculate total iterations for progress reporting
    total_iterations = max_iters

    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()  # Added for student accuracy tracking
        origin_accuracies = AverageMeter(
        )  # Added for teacher accuracy tracking

        for batch_idx, (data, target) in enumerate(train_loader):
            current_iter_in_epoch = iter_nums - start_epoch
            progress = (current_iter_in_epoch + 1) / total_iterations if total_iterations > 0 else 0

            iter_nums += 1
            if iter_nums > start_epoch + max_iters:
                finish = True
                break

            data_time.update(time.time() - end)

            # Sanitize inputs
            if isinstance(data, torch.Tensor):
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6,
                                        neginf=-1e6).to(device)
            else:
                data = safe_to_device(data, device)
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            target = target.to(device)

            # Apply mixup augmentation (only in phase 2 for stability)
            # For synthetic data, use stronger augmentation to prevent overfitting
            if phase == 2 and np.random.rand() < 0.7:  # Increased from 0.5
                data, _, _, _ = mixup_data(
                    data, target, alpha=0.3)  # Reduced alpha for more mixing

            # Add slight Gaussian noise to synthetic images for regularization
            if np.random.rand() < 0.3:
                noise = torch.randn_like(data) * 0.02
                data = data + noise
                data = torch.clamp(data, -3, 3)  # Clamp to reasonable range

            # Extract features for AMFR-CR and DGKD
            with torch.no_grad():
                teacher_features_dict, teacher_logits = extract_features_from_model(
                    origin_model, data, layer_names_teacher, device)
                # Build teacher outputs dict for AMFR-CR
                teacher_outputs_amfr = {
                    'features':
                    [teacher_features_dict[name] for name in layer_names_teacher[:3]],
                    'logits':
                    teacher_logits,
                    'final_features':
                    teacher_features_dict[layer_names_teacher[-1]]
                }
                # Extract features for DGKD
                teacher_features_dgkd = extract_features_for_dgkd(
                    origin_model, data, layer_names_teacher, device)


            student_features_dict, student_logits = extract_features_from_model(
                model, data, layer_names_student, device)
            # Build student outputs dict for AMFR-CR
            student_outputs_amfr = {
                'features':
                [student_features_dict[name] for name in layer_names_student],
                'logits':
                student_logits,
                'final_features':
                student_features_dict[layer_names_student[-1]]
            }
            # Extract features for DGKD
            student_features_dgkd = extract_features_for_dgkd(
                model, data, layer_names_student, device)


            # Compute AMFR-CR loss
            loss_dict_amfr = amfr_criterion(student_outputs_amfr, teacher_outputs_amfr,
                                            target, data)
            amfr_loss = loss_dict_amfr['total']

            # Compute DGKD loss
            loss_dict_dgkd = dgkd_criterion(student_features_dgkd, teacher_features_dgkd, target)
            dgkd_loss = loss_dict_dgkd['total']

            # Combine losses (example: weighted sum, adjust weights as needed)
            # For now, let's use a simple sum. Later, tuning will be required.
            loss = amfr_loss + dgkd_loss

            if not assert_finite("loss", loss):
                print(f"Skipping batch {iter_nums} due to non-finite loss")
                continue

            loss = loss / accumulation_steps
            losses.update(loss.data.item() * accumulation_steps, data.size(0))

            # Early stopping check for synthetic data
            current_loss = losses.avg
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(
                    f"\nEarly stopping at iteration {iter_nums}: no improvement for {patience_limit} iterations"
                )
                print(
                    f"Best loss: {best_loss:.4f}, Current loss: {current_loss:.4f}")
                finish = True
                break

            try:
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(filter(
                        lambda p: p.requires_grad, model.parameters()),
                                                   max_norm=5.0)
                    optimizer.step()

                    # Step the scheduler
                    scheduler.step()

                    optimizer.zero_grad()

                    # Update EMA model (if available in parent scope)
                    if 'ema_model' in locals() and ema_model is not None:
                        ema_model.update(model)

            except Exception as e:
                print(f"Backward/step failed at iteration {iter_nums}: {e}")
                if args.cuda:
                    torch.cuda.empty_cache()
                continue

            # Compute accuracies
            prec1, prec5 = accuracy(student_logits, target, topk=(1, 5))
            origin_prec1, origin_prec5 = accuracy(teacher_logits,
                                                  target,
                                                  topk=(1, 5))

            # Update metrics
            losses.update(loss.item() * accumulation_steps, data.size(0))
            accuracies.update(prec1.item(), data.size(0))
            origin_accuracies.update(origin_prec1.item(), data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if iter_nums % args.print_freq == 0:
                print(
                    f"Epoch [{current_iter_in_epoch+1}/{total_iterations}] "
                    f"Iter [{iter_nums}/{start_epoch+max_iters}] "
                    f"Progress: {progress*100:.2f}% | "
                    f"Total Loss: {losses.val:.4f} (Avg: {losses.avg:.4f}) | "
                    f"AMFR Loss: {amfr_loss.item():.4f} | "
                    f"DGKD Loss: {dgkd_loss.item():.4f} | "
                    f"Student Acc: {accuracies.val:.2f}% (Avg: {accuracies.avg:.2f}%) | "
                    f"Teacher Acc: {origin_accuracies.val:.2f}% (Avg: {origin_accuracies.avg:.2f}%) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # End of epoch training
        if finish:
            break

    # Clean up hooks and models
    del teacher_features_dict, student_features_dict
    del teacher_outputs_amfr, student_outputs_amfr
    del teacher_features_dgkd, student_features_dgkd
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


def train(train_loader,
          optimizer,
          model,
          origin_model,
          args,
          scheduler=None,
          warmup_epochs=0):
    """Wrapper for backward compatibility"""
    train_progressive(train_loader,
                      optimizer,
                      model,
                      origin_model,
                      args,
                      scheduler,
                      warmup_epochs,
                      max_epochs=args.epoch,
                      phase=1,
                      rm_blocks=[])


def nmse_loss(p, z):
    p_norm = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mean((p_norm - z_norm)**2)


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
            lambda_kd = 0.3
            mu_nmse = 0.4
            nu_cc = 0.2
        return lambda_ce, lambda_kd, mu_nmse, nu_cc

    # Extract features from pre-GAP layer
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"
    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    origin_model.to(device)
    origin_model.eval()

    model.train()

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
                data = data.to(device)
            else:
                data = safe_to_device(data, device)
            target = target.to(device)
            data_time.update(time.time() - end)

            with torch.no_grad():
                t_logits, t_features = origin_model(data)
                t_logits, _ = origin_model(data)  # Added to get t_logits

            # check teacher features
            if not assert_finite("t_features", t_features):
                print(
                    f"[batch {iter_nums}] skipping: teacher features non-finite"
                )
                continue

            s_logits, s_features = model(data)

            # Basic finiteness checks
            if not assert_finite("s_features", s_features):
                print(
                    f"[batch {iter_nums}] skipping: student features non-finite"
                )
                continue
            if not assert_finite("s_logits", s_logits):
                print(
                    f"[batch {iter_nums}] skipping: student logits non-finite")
                continue

            ce_loss = ce_criterion(s_logits, target)
            if not assert_finite("ce_loss", ce_loss):
                print(f"[batch {iter_nums}] skipping: ce_loss non-finite")
                continue

            # Soft target KD loss with temperature scaling
            t_soft = F.softmax(t_logits / temperature, dim=1)
            s_soft = F.log_softmax(s_logits / temperature, dim=1)
            kd_soft_loss = F.kl_div(s_soft, t_soft,
                                    reduction='batchmean') * (temperature**2)

            if not assert_finite("kd_soft_loss", kd_soft_loss):
                print(f"[batch {iter_nums}] skipping: kd_soft_loss non-finite")
                continue

            l_ins = nmse_loss(s_features, t_features)
            if problematic_classes is None:
                l_cla = nmse_loss(s_features.T, t_features.T)
                cc_s = class_correlation_matrix(s_features)
                cc_t = class_correlation_matrix(t_features)
                cc_loss = torch.mean((cc_s - cc_t)**2)
            else:
                l_cla, cc_loss = compute_focused_class_losses(
                    s_features, t_features, target, problematic_classes)

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
                torch.nn.utils.clip_grad_norm_(filter(
                    lambda p: p.requires_grad, model.parameters()),
                                               max_norm=5.0)
                optimizer.step()
            except RuntimeError as e:
                print(f"[batch {iter_nums}] backward failed: {e}")
                # optionally save offending batch for inspection
                try:
                    torch.save(
                        {
                            "data": data.detach().cpu(),
                            "target": target.detach().cpu()
                        },
                        f"bad_batch_{iter_nums}.pt",
                    )
                    print(f"Saved bad batch bad_batch_{iter_nums}.pt")
                except Exception:
                    pass
                if args.cuda:
                    torch.cuda.empty_cache()
                continue

            losses.update(total_loss.item(), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if iter_nums % 50 == 0:
                print(f"Train: [{iter_nums}/{args.epoch}]\t"
                      f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})")


def compute_focused_class_losses(s_features, t_features, targets,
                                 focused_classes):
    """
    Compute class NMSE and correlation loss for selected classes only.
    """
    # Flatten features to [B, C]
    s_feat = s_features.view(s_features.size(0), -1).contiguous()
    t_feat = t_features.view(t_features.size(0), -1).contiguous()

    # Mask for focused classes
    mask = torch.tensor([c in focused_classes for c in targets.cpu().tolist()],
                        device=targets.device)
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
    cc_loss = torch.mean((cc_s - cc_t)**2)

    return class_nmse, cc_loss


def metric(args, metric_loader, model, origin_model, trained=False):
    criterion = torch.nn.MSELoss(reduction="mean")

    # switch to train mode
    device = torch.device("cuda" if args.cuda else "cpu")
    origin_model.to(device)
    model.to(device)
    origin_model.eval()

    model.eval()
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    origin_accuracies = AverageMeter()

    # Initialize per-category accuracy tracking
    num_classes = 1000  # Assuming ImageNet classes
    if args.cuda:
        correct_per_class = torch.zeros(num_classes).cuda()
        total_per_class = torch.zeros(num_classes).cuda()
        origin_correct_per_class = torch.zeros(num_classes).cuda()
        # Initialize per-class MSE loss tracking
        mse_loss_per_class = torch.zeros(num_classes).cuda()

    end = time.time()
    for i, (data, target) in enumerate(metric_loader):
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)
            loss = criterion(s_features, t_features)

            # Calculate overall accuracy
            acc = accuracy(s_output, target, topk=(1, ))[0]
            origin_acc = accuracy(t_output, target, topk=(1, ))[0]

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
                ))

    print(" * Metric Loss {loss.avg:.4f}".format(loss=losses))

    problematic_classes = []
    print(
        f"Overall Accuracy - Pruned: {accuracies.avg:.2f}% (Teacher: {origin_accuracies.avg:.2f}%)"
    )

    return losses.avg, accuracies.avg, origin_accuracies.avg, problematic_classes


def train_focused(train_loader, metric_loader, optimizer, model, origin_model,
                  args, target_classes):
    """
    Train the model focusing on specific classes with large accuracy differences
    """
    criterion = torch.nn.MSELoss(reduction="mean")
    cls_criterion = torch.nn.CrossEntropyLoss()

    # switch to train mode
    device = torch.device("cuda" if args.cuda else "cpu")
    origin_model.to(device)
    model.to(device)

    origin_model.eval()

    model.train()

    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"
    if args.cuda:
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
            mask = torch.tensor([t in target_classes for t in target],
                                device=data.device)
            if not mask.any():
                continue
            data = data[mask].to(device)
            target = target[mask].to(device)

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
                print("Focused Train: [{0}/{1}]\t"
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "Loss {losses.val:.4f} ({losses.avg:.4f})".format(
                          iter_nums,
                          args.epoch,
                          batch_time=batch_time,
                          data_time=data_time,
                          losses=losses,
                      ))

            if iter_nums % 400 == 0:
                _, acc, _, _ = metric(args, metric_loader, model, origin_model)
                print(f"Current Accuracy: {acc:.4f}")
                model.train()

    return model


def log_model_parameters(model,
                         model_name="Model",
                         log_file="model_parameters.log"):
    # Check for existing files and increment the file name if necessary
    base_name, ext = os.path.splitext(log_file)
    counter = 1
    while os.path.exists(log_file):
        log_file = f"{base_name}_{counter}{ext}"
        counter += 1

    # Set PyTorch print options to display all elements
    torch.set_printoptions(edgeitems=None,
                           linewidth=1000,
                           sci_mode=False,
                           threshold=float("inf"))

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
    torch.set_printoptions(edgeitems=3,
                           linewidth=80,
                           sci_mode=None,
                           threshold=1000)

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
    # analysis_results = analyze_batchnorm_gamma_distribution(model, save_plot=True) # This function is not defined in the provided code. Assuming it's a placeholder or intended to be implemented.

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

    # return gamma_values, analysis_results # Returning analysis_results which might not be defined.
    return gamma_values  # Returning only gamma_values as analysis_results is not defined.