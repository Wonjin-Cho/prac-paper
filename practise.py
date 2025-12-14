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
from novel_method import EnhancedContrastiveTrainer, train_with_enhanced_contrastive


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
from novel_method import EnhancedContrastiveTrainer, train_with_enhanced_contrastive


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
    # Use frequency-adaptive training
    train_with_frequency_filtering(train_loader, optimizer, prune_model, origin_model, args, scheduler, warmup_epochs)
    print(
        "compute recoverability {} takes {}s".format(
            rm_blocks, time.time() - recover_time


def train_with_frequency_filtering(train_loader, optimizer, model, origin_model, args, scheduler=None, warmup_epochs=0):
    """Training with frequency domain filtering and adaptive augmentation"""
    from frequency_filter import FrequencyFilter, AdaptiveNoiseInjection, SpectralNormalization, compute_difficulty_score
    
    end = time.time()
    criterion = torch.nn.MSELoss(reduction="mean")
    noise_injector = AdaptiveNoiseInjection(noise_std=0.03)

    # Switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.eval()
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"

    accumulation_steps = 2
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
            
            data_time.update(time.time() - end)
            
            # Sanitize inputs
            if isinstance(data, torch.Tensor):
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6).cuda()
            else:
                data = safe_to_device(data)
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

            # Get teacher features first
            with torch.no_grad():
                t_output, t_features = origin_model(data)
                
                # Compute difficulty score
                s_output_temp, _ = model(data)
                difficulty = compute_difficulty_score(s_output_temp, t_output, target)

            # Apply frequency filtering based on training progress
            progress = iter_nums / args.epoch
            
            if progress < 0.3:
                # Early stage: strong low-pass filtering for stability
                filtered_data = FrequencyFilter.low_pass_filter(data, cutoff_ratio=0.5)
            elif progress < 0.7:
                # Mid stage: adaptive mixing
                filtered_data = FrequencyFilter.adaptive_frequency_mix(data, t_features, alpha=0.6)
            else:
                # Late stage: spectral normalization + light filtering
                filtered_data = SpectralNormalization.spectral_normalize(data)
                filtered_data = FrequencyFilter.low_pass_filter(filtered_data, cutoff_ratio=0.7)
            
            # Add adaptive noise for robustness
            augmented_data = noise_injector(filtered_data, difficulty)
            
            # Check filtered data
            if not assert_finite("augmented_data", augmented_data):
                print("Skipping batch due to non-finite filtered data")
                continue

            # Get teacher features with filtered data
            with torch.no_grad():
                _, t_features_filtered = origin_model(augmented_data)
                if not assert_finite("t_features_filtered", t_features_filtered):
                    print("Skipping batch due to non-finite teacher features")
                    continue

            # Student forward with filtered data
            output, s_features = model(augmented_data)
            
            if not assert_finite("s_features", s_features):
                print("Skipping batch due to non-finite student features")
                continue

            # Multi-scale feature loss
            loss = criterion(s_features, t_features_filtered)
            
            # Add original feature alignment loss with lower weight
            with torch.no_grad():
                _, t_features_orig = origin_model(data)
            _, s_features_orig = model(data)
            loss_orig = criterion(s_features_orig, t_features_orig)
            
            # Combined loss: filtered (0.7) + original (0.3)
            total_loss = 0.7 * loss + 0.3 * loss_orig
            
            if not assert_finite("total_loss", total_loss):
                print("Skipping batch due to non-finite loss")
                continue

            # Normalize loss by accumulation steps
            total_loss = total_loss / accumulation_steps
            losses.update(total_loss.data.item() * accumulation_steps, augmented_data.size(0))

            try:
                total_loss.backward()

                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    # Learning rate warmup and scheduling
                    if warmup_epochs > 0 and iter_nums <= warmup_epochs:
                        lr = args.lr * (iter_nums / warmup_epochs)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    elif scheduler is not None and iter_nums > warmup_epochs:
                        scheduler.step()
            except Exception as e:
                print("Backward/step failed:", e)
                torch.cuda.empty_cache()
                continue
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if iter_nums % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    "FreqTrain: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    "LR {lr:.6f}\t"
                    "Progress {prog:.1%}".format(
                        iter_nums,
                        args.epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                        lr=current_lr,
                        prog=progress,
                    )
                )

        )
    )


def train(train_loader, optimizer, model, origin_model, args, scheduler=None, warmup_epochs=0):
    # Data loading code
    end = time.time()
    criterion = torch.nn.MSELoss(reduction="mean")

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.eval()
    model.get_feat = "pre_GAP"
    origin_model.get_feat = "pre_GAP"

    # Gradient accumulation
    accumulation_steps = 2

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
            # measure data loading time
            data_time.update(time.time() - end)
            # sanitize inputs
            if isinstance(data, torch.Tensor):
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6).cuda()
            else:
                data = safe_to_device(data)
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

            with torch.no_grad():
                t_output, t_features = origin_model(data)

            # check teacher features
            if not assert_finite("t_features", t_features):
                print("Skipping batch due to non-finite teacher features")
                continue

            output, s_features = model(data)

            # check student features
            if not assert_finite("s_features", s_features):
                print("Skipping batch due to non-finite student features")
                continue

            loss = criterion(s_features, t_features)
            if not assert_finite("loss", loss):
                print("Skipping batch due to non-finite loss")
                continue

            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            losses.update(loss.data.item() * accumulation_steps, data.size(0))

            try:
                loss.backward()

                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    # Learning rate warmup and scheduling
                    if warmup_epochs > 0 and iter_nums <= warmup_epochs:
                        lr = args.lr * (iter_nums / warmup_epochs)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    elif scheduler is not None and iter_nums > warmup_epochs:
                        scheduler.step()
            except Exception as e:
                print("Backward/step failed:", e)
                torch.cuda.empty_cache()
                continue
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_nums % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    "Train: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    "LR {lr:.6f}".format(
                        iter_nums,
                        args.epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                        lr=current_lr,
                    )
                )

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
                t_logits, t_features = origin_model(data)

            # check teacher features
            if not assert_finite("t_features", t_features):
                print(f"[batch {iter_nums}] skipping: teacher features non-finite")
                continue

            s_logits, s_features = model(data)
            # check student features
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

    print(" * Metric Loss {loss.avg:.4f}")

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