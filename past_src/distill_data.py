"""
SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning (ICLR 2025)

Authors:
- Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
- Jongjin Kim (j2kim99@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.1

Date : Feb. 7th, 2025

Main Contact: Minjun Kim

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

distill_data.py
    - codes for generating distilled data

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""

import gc
import os
import sys
import time
import pickle
import random
import logging

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.functional import conv2d

from torchvision import transforms

import builtins


def check_path(model_path):
    """
    Check if the directory exists, if not create it.
    Args:
        model_path: path to the model
    """
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# def apply_unsharp_masking(image, amount=1.0, radius=1.0):
#     """
#     Apply unsharp masking to enhance image sharpness.
#     Args:
#         image: input image tensor of shape (B, C, H, W)
#         amount: amount of sharpening (default: 1.0)
#         radius: radius of the Gaussian blur (default: 1.0)
#     Returns:
#         sharpened: sharpened image tensor
#     """
#     # Create Gaussian kernel
#     kernel_size = int(2 * radius + 1)
#     sigma = radius / 3.0
#     x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=image.device)
#     x = x.view(-1, 1)
#     y = x.view(1, -1)
#     kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
#     kernel = kernel / kernel.sum()
#     kernel = kernel.view(1, 1, kernel_size, kernel_size)
#     kernel = kernel.repeat(image.size(1), 1, 1, 1)

#     # Apply Gaussian blur
#     padding = kernel_size // 2
#     blurred = conv2d(image, kernel, padding=padding, groups=image.size(1))

#     # Calculate high-pass component
#     high_pass = image - blurred

#     # Apply sharpening
#     sharpened = image + amount * high_pass

#     # Clamp values to valid range
#     sharpened = torch.clamp(sharpened, 0, 1)

#     return sharpened

# def get_initial_spatial_size(model_name):
#     """
#     Get the initial spatial size for different models.
#     Args:
#         model_name: name of the model
#     Returns:
#         tuple: (height, width) of initial spatial size
#     """
#     if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
#         return (32, 32)
#     else:
#         return (224, 224)

# def is_before_downsampling(spatial_size, initial_size):
#     """
#     Check if the current spatial size is before any downsampling.
#     Args:
#         spatial_size: current spatial size (H, W)
#         initial_size: initial spatial size (H, W)
#     Returns:
#         bool: True if before downsampling
#     """
#     return spatial_size[0] >= initial_size[0] and spatial_size[1] >= initial_size[1]


def generate_calib_centers(args, teacher_model, beta_ce=5):
    """
    Generate calibration centers for the teacher model.
    Args:
        args: arguments
        teacher_model: teacher model
        beta_ce: beta for cross entropy loss
    Returns:
        refined_gaussian: refined gaussian data
    """
    calib_path = os.path.join(
        args.save_path_head, args.model + "_calib_centers" + ".pickle"
    )
    if not os.path.exists(calib_path):
        model_name = args.model

        if model_name == "resnet20_cifar10":
            num_classes = 10
        elif model_name == "resnet20_cifar100":
            num_classes = 100
        elif model_name == "resnet34_cifar100":
            num_classes = 100
        else:
            num_classes = 1000

        if model_name in ["resnet20_cifar10", "resnet20_cifar100", "resnet34_cifar100"]:
            shape = (args.batch_size, 3, 32, 32)
        else:
            shape = (args.batch_size, 3, 224, 224)

        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        refined_gaussian = []

        ce_loss = nn.CrossEntropyLoss(reduction="none").cuda()
        mse_loss = nn.MSELoss().cuda()

        mean_list = []
        var_list = []
        teacher_running_mean = []
        teacher_running_var = []

        def hook_fn_forward(module, _input, output):
            _input = _input[0]
            mean = _input.mean([0, 2, 3])
            var = _input.var([0, 2, 3], unbiased=False)

            mean_list.append(mean)
            var_list.append(var)
            teacher_running_mean.append(module.running_mean)
            teacher_running_var.append(module.running_var)

        for _, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(hook_fn_forward)

        total_time = time.time()

        epsilon = 1e-6  # Small value to prevent division by zero/NaN

        for i in range(num_classes // args.batch_size + 1):
            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True
            optimizer = optim.Adam([gaussian_data], lr=0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, min_lr=0.05, verbose=False, patience=50
            )

            if (i + 1) * args.batch_size <= num_classes:
                labels = torch.tensor(
                    [i * args.batch_size + j for j in range(args.batch_size)],
                    dtype=torch.long,
                    device="cuda",
                )
            else:
                labels = torch.tensor(
                    [
                        i * args.batch_size + j
                        for j in range(num_classes % args.batch_size)
                    ],
                    dtype=torch.long,
                    device="cuda",
                )

            if len(labels) < args.batch_size:
                labels = torch.nn.functional.pad(
                    labels, (0, args.batch_size - len(labels))
                )

            batch_time = time.time()
            for it in range(1500):
                new_gaussian_data = []
                for j, jth_data in enumerate(gaussian_data):
                    new_gaussian_data.append(jth_data)
                new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                # Apply unsharp masking
                # new_gaussian_data = apply_unsharp_masking(new_gaussian_data, amount=0.5, radius=1.0)

                mean_list.clear()
                var_list.clear()
                teacher_running_mean.clear()
                teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                loss_target = beta_ce * (ce_loss(output, labels)).mean()

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for n, nth_mean in enumerate(mean_list):
                    if n < (len(mean_list) + 2) // 2 - 2:
                        mean_loss += 0.2 * mse_loss(
                            nth_mean, teacher_running_mean[n].detach()
                        )
                        var_loss += 0.2 * mse_loss(
                            var_list[n] + epsilon,
                            teacher_running_var[n].detach() + epsilon,
                        )
                    else:
                        mean_loss += 1.1 * mse_loss(
                            nth_mean, teacher_running_mean[n].detach()
                        )
                        var_loss += 1.1 * mse_loss(
                            var_list[n] + epsilon,
                            teacher_running_var[n].detach() + epsilon,
                        )

                mean_loss = mean_loss / len(mean_list)
                var_loss = var_loss / len(mean_list)

                total_loss = mean_loss + var_loss + loss_target

                print(
                    i,
                    it,
                    "lr",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    "mean_loss",
                    mean_loss.item(),
                    "var_loss",
                    var_loss.item(),
                    "loss_target",
                    loss_target.item(),
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                gaussian_data.data = torch.clamp(gaussian_data.data, -3, 3)
                scheduler.step(total_loss.item())

            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == labels)
                # print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())

            print(
                f"Time for {i} batch for {it} iters: {time.time() - batch_time:.2f} sec."
            )

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        print(
            f"Total time for {num_classes // args.batch_size} "
            f"batches: {time.time() - total_time:.2f} sec."
        )
        check_path(calib_path)
        with open(calib_path, "wb") as fp:
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        del refined_gaussian

    with open(calib_path, "rb") as f:
        refined_gaussian = pickle.load(f)
    return refined_gaussian


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Forward pass for the LabelSmoothing module.
        Args:
            x: input tensor
            target: target tensor
        Returns:
            loss: loss value
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


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

    def clear(self):
        """
        Clear the output.
        """
        self.outputs = None


class DistillData:
    """
    Construct the distilled data.
    Args:
        args: arguments
    """

    def __init__(self, args):
        LOG = logging.getLogger("main")
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.args = args

        if args.lbns:
            self.calib_centers = args.calib_centers
            self.calib_running_mean = []
            self.calib_running_var = []
            self.calib_data_means = []
            self.calib_data_vars = []

    def hook_fn_forward(self, module, _input, _):
        """
        Forward hook function for the batch normalization layer.
        Args:
            module: module
            _input: input
            output: output
        """
        _input = _input[0]
        mean = _input.mean([0, 2, 3])
        var = _input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def get_distil_data(
        self,
        model_name="resnet50",
        teacher_model=None,
        batch_size=256,
        num_batch=1,
        group=1,
        aug_margin=0.4,
        beta=1.0,
        gamma=0,
        save_path_head="",
        target_class=None,
        problematic_labels=None,
    ):
        """
        Generate the distilled data.
        Args:
            model_name: model name
            teacher_model: teacher model
            batch_size: batch size
            num_batch: number of batches
            group: group
            aug_margin: augmentation margin
            beta: beta
            gamma: gamma
            save_path_head: save path head
            target_class: if specified, generate data only for this class
            problematic_labels: list or set of problematic labels (for per-batch iters)"""

        if target_class is not None:
            data_path = os.path.join(
                save_path_head, f"class_{target_class}_data.pickle"
            )
            label_path = os.path.join(
                save_path_head, f"class_{target_class}_labels.pickle"
            )
        else:
            data_path = os.path.join(
                save_path_head,
                model_name
                + "_refined_gaussian_hardsample_"
                + "beta"
                + str(beta)
                + "_gamma"
                + str(gamma)
                + "_group"
                + str(group)
                + ".pickle",
            )
            label_path = os.path.join(
                save_path_head,
                model_name
                + "_labels_hardsample_"
                + "beta"
                + str(beta)
                + "_gamma"
                + str(gamma)
                + "_group"
                + str(group)
                + ".pickle",
            )

        print(data_path, label_path)

        check_path(data_path)
        check_path(label_path)

        if model_name == "resnet20_cifar10":
            self.num_classes = 10
        elif model_name == "resnet20_cifar100":
            self.num_classes = 100
        elif model_name == "resnet34_cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 1000

        if model_name in ["resnet20_cifar10", "resnet20_cifar100", "resnet34_cifar100"]:
            shape = (batch_size, 3, 32, 32)
        else:
            shape = (batch_size, 3, 224, 224)

        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        refined_gaussian = []
        labels_list = []

        ce_loss = nn.CrossEntropyLoss(reduction="none").cuda()
        mse_loss = nn.MSELoss().cuda()

        for _, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

        total_time = time.time()

        # Generate data for single batch
        if model_name in ["resnet20_cifar10", "resnet20_cifar100", "resnet34_cifar100"]:
            rrc = transforms.RandomResizedCrop(size=32, scale=(aug_margin, 1.0))
        else:
            rrc = transforms.RandomResizedCrop(size=224, scale=(aug_margin, 1.0))
        rhf = transforms.RandomHorizontalFlip()

        # # Generate random RGB data
        # gaussian_data = torch.randn(shape).cuda()
        # gaussian_data.requires_grad = True

        # optimizer = optim.Adam([gaussian_data], lr=0.5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                  min_lr=1e-4,
        #                                                  verbose=False,
        #                                                  patience=50)

        # # If target_class is specified, use it for all samples
        # if target_class is not None:
        #     labels = torch.full((len(gaussian_data),), target_class, dtype=torch.long).cuda()
        # else:
        #     labels = torch.randint(0, self.num_classes, (len(gaussian_data),)).cuda()

        # gt = labels.data.detach().cpu().numpy()

        # # Determine iters based on problematic labels

        # if problematic_labels is not None:
        #     # If any label in this batch is problematic, use 2000 iters
        #     if any(l.item() in problematic_labels for l in labels):
        #         iters = 2000

        # batch_time = time.time()
        # for it in range(iters):
        #     if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
        #         new_gaussian_data = []
        #         for j, jth_data in enumerate(gaussian_data):
        #             new_gaussian_data.append(jth_data)
        #         new_gaussian_data = torch.stack(new_gaussian_data).cuda()
        #     else:
        #         if random.random() < 0.5:
        #             new_gaussian_data = []
        #             for j, jth_data in enumerate(gaussian_data):
        #                 new_gaussian_data.append(rhf(rrc(jth_data)))
        #             new_gaussian_data = torch.stack(new_gaussian_data).cuda()
        #         else:
        #             new_gaussian_data = []
        #             for j, jth_data in enumerate(gaussian_data):
        #                 new_gaussian_data.append(jth_data)
        #             new_gaussian_data = torch.stack(new_gaussian_data).cuda()

        #     self.mean_list.clear()
        #     self.var_list.clear()
        #     self.teacher_running_mean.clear()
        #     self.teacher_running_var.clear()

        #     output = teacher_model(new_gaussian_data)
        #     # Handle tuple output (logits, features)
        #     if isinstance(output, tuple):
        #         logits = output[0]
        #     else:
        #         logits = output

        #     d_acc = np.mean(np.argmax(logits.detach().cpu().numpy(), axis=1) == gt)
        #     a = F.softmax(logits, dim=1)
        #     ###############################
        # Parameters
        easy_threshold = 0.9
        assert batch_size in [32, 64], "Batch size must be 32 or 64"
        max_samples = 1280
        total_collected = 0
        iters = 2000
        epsilon = 1e-6  # Small value to prevent division by zero/NaN
        easy_gaussians = []
        easy_labels = []

        while total_collected < max_samples:
            # Initialize synthetic batch
            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True

            optimizer = optim.Adam([gaussian_data], lr=0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, min_lr=1e-4, verbose=False, patience=50
            )

            if target_class is not None:
                labels = torch.full(
                    (len(gaussian_data),), target_class, dtype=torch.long
                ).cuda()
            else:
                labels = torch.randint(
                    0, self.num_classes, (len(gaussian_data),)
                ).cuda()

            gt = labels.data.detach().cpu().numpy()

            for it in range(iters):  # original fixed iteration loop
                if model_name in [
                    "resnet20_cifar10",
                    "resnet20_cifar100",
                    "resnet34_cifar100",
                ]:
                    new_gaussian_data = gaussian_data.clone()
                else:
                    if random.random() < 0.5:
                        new_gaussian_data = torch.stack(
                            [rhf(rrc(img)) for img in gaussian_data]
                        ).cuda()
                    else:
                        new_gaussian_data = gaussian_data.clone()

                self.mean_list.clear()
                self.var_list.clear()
                self.teacher_running_mean.clear()
                self.teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                logits = output[0] if isinstance(output, tuple) else output
                a = F.softmax(logits, dim=1)

                mask = torch.zeros_like(a)
                b = labels.unsqueeze(1)
                mask = mask.scatter_(1, b, torch.ones_like(b).float())
                p = a[mask.bool()]

                loss_target = (
                    beta * ((1 - p).pow(gamma) * ce_loss(logits, labels)).mean()
                )

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for n, nth_mean in enumerate(self.mean_list):
                    mean_loss += mse_loss(
                        nth_mean.cpu(), self.teacher_running_mean[n].detach().cpu()
                    )
                    var_loss += mse_loss(
                        self.var_list[n].cpu() + epsilon,
                        self.teacher_running_var[n].detach().cpu() + epsilon,
                    )

                mean_loss /= len(self.mean_list)
                var_loss /= len(self.mean_list)

                total_loss = mean_loss + var_loss + loss_target

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                gaussian_data.data = torch.clamp(gaussian_data.data, -3, 3)
                scheduler.step(total_loss.item())

            # After optimization, check if samples are easy
            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                logits = output[0] if isinstance(output, tuple) else output
                softmax_scores = F.softmax(logits, dim=1)
                target_scores = softmax_scores.gather(1, labels.unsqueeze(1)).squeeze(1)

                # Instead of accepting by confidence, accept/reject by how close
                # the batch statistics (per-BN-layer mean/var) are to the teacher's running stats.
                # Compute per-layer average absolute differences for mean and var and aggregate.
                stat_threshold = getattr(self.args, "stat_threshold", 0.1)
                stat_diffs = []
                try:
                    # use the collected self.mean_list / self.var_list and running stats
                    for n, batch_mean in enumerate(self.mean_list):
                        batch_var = self.var_list[n]
                        running_mean = self.teacher_running_mean[n].detach()
                        running_var = self.teacher_running_var[n].detach()

                        # move running stats to same device as batch stats if needed
                        if running_mean.device != batch_mean.device:
                            running_mean = running_mean.to(batch_mean.device)
                        if running_var.device != batch_var.device:
                            running_var = running_var.to(batch_var.device)

                        mean_diff = (batch_mean - running_mean).abs().mean().item()
                        var_diff = (batch_var - running_var).abs().mean().item()
                        stat_diffs.append(mean_diff + var_diff)

                    stat_score = (
                        float(np.mean(stat_diffs))
                        if len(stat_diffs) > 0
                        else float("inf")
                    )
                except Exception as e:
                    # If statistics couldn't be computed, reject the batch and log
                    print(f"[!] Failed to compute batch-statistics diff: {e}")
                    stat_score = float("inf")

                # Accept if stat_score below threshold (smaller -> closer to running stats)
                if stat_score < stat_threshold:
                    print(
                        f"[✓] Accepted batch #{total_collected // batch_size + 1} | stat_score: {stat_score:.6f}"
                    )
                    easy_gaussians.append(gaussian_data.detach().cpu())
                    easy_labels.append(labels.detach().cpu())
                    total_collected += batch_size
                else:
                    print(
                        f"[✗] Rejected batch | stat_score: {stat_score:.6f} (threshold {stat_threshold})"
                    )

            del gaussian_data, optimizer, scheduler, labels
            torch.cuda.empty_cache()

        print(f"Total time: {time.time() - total_time:.2f} sec.")

        final_data = torch.cat(easy_gaussians, dim=0).numpy()[:max_samples]
        final_labels = torch.cat(easy_labels, dim=0).numpy()[:max_samples]

        with open(data_path, "wb") as fp:
            pickle.dump([final_data], fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:
            pickle.dump([final_labels], fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(data_path, label_path)

        return final_data, final_labels

    # def get_distil_data(self, model_name="resnet50", teacher_model=None, batch_size=256,
    #                   num_batch=1, group=1, aug_margin=0.4, beta=1.0, gamma=0, save_path_head="", target_class=None, problematic_labels=None):

    #     Generate the distilled data.
    #     Args:
    #         model_name: model name
    #         teacher_model: teacher model
    #         batch_size: batch size
    #         num_batch: number of batches
    #         group: group
    #         aug_margin: augmentation margin
    #         beta: beta
    #         gamma: gamma
    #         save_path_head: save path head
    #         target_class: if specified, generate data only for this class
    #         problematic_labels: list or set of problematic labels (for per-batch iters)

    #     if target_class is not None:
    #         data_path = os.path.join(save_path_head, f"class_{target_class}_data.pickle")
    #         label_path = os.path.join(save_path_head, f"class_{target_class}_labels.pickle")
    #     else:
    #         data_path = os.path.join(save_path_head, model_name+"_refined_gaussian_hardsample_" \
    #                     + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")
    #         label_path = os.path.join(save_path_head, model_name+"_labels_hardsample_" \
    #                     + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")

    #     print(data_path, label_path)

    #     check_path(data_path)
    #     check_path(label_path)

    #     if model_name == 'resnet20_cifar10':
    #         self.num_classes = 10
    #     elif model_name == 'resnet20_cifar100':
    #         self.num_classes = 100
    #     elif model_name == 'resnet34_cifar100':
    #         self.num_classes = 100
    #     else:
    #         self.num_classes = 1000

    #     if model_name in ['resnet20_cifar10','resnet20_cifar100', 'resnet34_cifar100']:
    #         shape = (batch_size, 3, 32, 32)
    #     else:
    #         shape = (batch_size, 3, 224, 224)

    #     teacher_model = teacher_model.cuda()
    #     teacher_model = teacher_model.eval()

    #     refined_gaussian = []
    #     labels_list = []

    #     ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
    #     mse_loss = nn.MSELoss().cuda()

    #     for _, m in teacher_model.named_modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.register_forward_hook(self.hook_fn_forward)

    #     total_time = time.time()

    #     # Generate data for single batch
    #     if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
    #         rrc = transforms.RandomResizedCrop(size=32,scale=(aug_margin, 1.0))
    #     else:
    #         rrc = transforms.RandomResizedCrop(size=224,scale=(aug_margin, 1.0))
    #     rhf = transforms.RandomHorizontalFlip()

    #     # Generate random RGB data
    #     gaussian_data = torch.randn(shape).cuda()
    #     gaussian_data.requires_grad = True

    #     optimizer = optim.Adam([gaussian_data], lr=0.5)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                      min_lr=1e-4,
    #                                                      verbose=False,
    #                                                      patience=50)

    #     # If target_class is specified, use it for all samples
    #     if target_class is not None:
    #         labels = torch.full((len(gaussian_data),), target_class, dtype=torch.long).cuda()
    #     else:
    #         labels = torch.randint(0, self.num_classes, (len(gaussian_data),)).cuda()

    #     gt = labels.data.detach().cpu().numpy()

    #     # Determine iters based on problematic labels
    #     iters = 500
    #     if problematic_labels is not None:
    #         # If any label in this batch is problematic, use 2000 iters
    #         if any(l.item() in problematic_labels for l in labels):
    #             iters = 2000

    #     epsilon = 1e-6  # Small value to prevent division by zero/NaN

    #     batch_time = time.time()
    #     for it in range(iters):
    #         if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
    #             new_gaussian_data = []
    #             for j, jth_data in enumerate(gaussian_data):
    #                 new_gaussian_data.append(jth_data)
    #             new_gaussian_data = torch.stack(new_gaussian_data).cuda()
    #         else:
    #             if random.random() < 0.5:
    #                 new_gaussian_data = []
    #                 for j, jth_data in enumerate(gaussian_data):
    #                     new_gaussian_data.append(rhf(rrc(jth_data)))
    #                 new_gaussian_data = torch.stack(new_gaussian_data).cuda()
    #             else:
    #                 new_gaussian_data = []
    #                 for j, jth_data in enumerate(gaussian_data):
    #                     new_gaussian_data.append(jth_data)
    #                 new_gaussian_data = torch.stack(new_gaussian_data).cuda()

    #         self.mean_list.clear()
    #         self.var_list.clear()
    #         self.teacher_running_mean.clear()
    #         self.teacher_running_var.clear()

    #         output = teacher_model(new_gaussian_data)
    #         # Handle tuple output (logits, features)
    #         if isinstance(output, tuple):
    #             logits = output[0]
    #         else:
    #             logits = output

    #         d_acc = np.mean(np.argmax(logits.detach().cpu().numpy(), axis=1) == gt)
    #         a = F.softmax(logits, dim=1)
    #         ###############################

    #         mask = torch.zeros_like(a)
    #         b=labels.unsqueeze(1)
    #         mask=mask.scatter_(1,b,torch.ones_like(b).float())
    #         p=a[mask.bool()]

    #         loss_target = beta * ((1-p).pow(gamma) * ce_loss(logits, labels)).mean()

    #         mean_loss = torch.zeros(1).cuda()
    #         var_loss = torch.zeros(1).cuda()
    #         for n, nth_mean in enumerate(self.mean_list):
    #             mean_loss += mse_loss(nth_mean.cpu(),
    #                                   self.teacher_running_mean[n].detach().cpu())
    #             # Add epsilon to variance for stability
    #             var_loss += mse_loss(self.var_list[n].cpu() + epsilon,
    #                                 self.teacher_running_var[n].detach().cpu() + epsilon)

    #         mean_loss = mean_loss / len(self.mean_list)
    #         var_loss = var_loss / len(self.mean_list)

    #         total_loss = mean_loss + var_loss + loss_target

    #         print(it, 'lr', optimizer.state_dict()['param_groups'][0]['lr'],
    #                 'mean_loss', mean_loss.item(), 'var_loss',
    #                 var_loss.item(), 'loss_target', loss_target.item())

    #         optimizer.zero_grad()
    #         total_loss.backward()
    #         optimizer.step()
    #         # Clamp synthetic data to a reasonable range after update
    #         gaussian_data.data = torch.clamp(gaussian_data.data, -3, 3)
    #         scheduler.step(total_loss.item())

    #     with torch.no_grad():
    #         output = teacher_model(gaussian_data.detach())
    #         # Handle tuple output (logits, features)
    #         if isinstance(output, tuple):
    #             logits = output[0]
    #         else:
    #             logits = output
    #         d_acc = np.mean(np.argmax(logits.detach().cpu().numpy(), axis=1) == gt)
    #         # print('d_acc', d_acc)

    #     refined_gaussian.append(gaussian_data.detach().cpu().numpy())
    #     labels_list.append(labels.detach().cpu().numpy())

    #     print(f"Time for generation: {time.time()-batch_time:.2f} sec.")

    #     gaussian_data = gaussian_data.cpu()
    #     del gaussian_data
    #     del optimizer
    #     del scheduler
    #     del labels
    #     torch.cuda.empty_cache()

    #     print(f"Total time: {time.time()-total_time:.2f} sec.")

    #     # Save the generated data
    #     with open(data_path, "wb") as fp:
    #         pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #     with open(label_path, "wb") as fp:
    #         pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #     return refined_gaussian[0], labels_list[0]  # Return both data and labels
