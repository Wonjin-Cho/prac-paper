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

generate_data.py
    - codes for generating synthetic data for zero-shot quantization

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""

import os
import sys
import argparse

import torch

from .distill_data import generate_calib_centers, DistillData

from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.models as models


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# model settings
def arg_parse():
    """
    Parses arguments for data generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34_cifar100",
            "resnet34",
            "resnet50",
            "mobilenet_w1",
            "mobilenetv2_w1",
            "shufflenet_g1_w1",
            "resnet20_cifar10",
            "resnet20_cifar100",
            "regnetx_600m",
        ],
        help="model to be quantized",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of distilled data"
    )
    parser.add_argument(
        "--num_data", type=int, default=1280, help="batch size of test data"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size of test data"
    )
    parser.add_argument("--group", type=int, default=1, help="group of generated data")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")
    parser.add_argument("--gamma", type=float, default=0.0, help="gamma")
    parser.add_argument("--save_path_head", type=str, default="", help="save_path_head")
    parser.add_argument("--radius", type=float, default=0.05, metavar="radius")
    parser.add_argument("--lbns", type=bool, default=False, metavar="lbns")
    parser.add_argument("--fft", type=bool, default=False, metavar="fft")

    arguments = parser.parse_args()

    if arguments.lbns:
        arguments.save_path_head = arguments.save_path_head + "_lbns"

    if arguments.fft:
        arguments.save_path_head = arguments.save_path_head + "_fft"
    return arguments


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if args.model == "resnet34_cifar100":
        module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "utils")
        )
        if module_path not in sys.path:
            sys.path.append(module_path)
        from get_resnet34 import resnet34_get_model

        model = resnet34_get_model()
        model.load_state_dict(torch.load("../checkpoints/resnet34.pth"))
        print("****** Full precision model loaded ******")
    # elif args.model == 'resnet34':
    #     model = models.resnet34(pretrained=True)
    #     print('****** Full precision model loaded ******')
    else:
        model = ptcv_get_model(args.model, pretrained=True)
        print("****** Full precision model loaded ******")

    if args.lbns:
        args.calib_centers = generate_calib_centers(args, model.cuda())

    DD = DistillData(args)
    dataloader = DD.get_distil_data(
        model_name=args.model,
        teacher_model=model.cuda(),
        batch_size=args.batch_size,
        group=args.group,
        beta=args.beta,
        gamma=args.gamma,
        save_path_head=args.save_path_head,
    )

    print("****** Data Generated ******")
