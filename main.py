import os
import argparse
import logging
import collections
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import pickle
from torch.utils.data import Dataset, DataLoader

import time
import torch.optim as optim
import torch.nn.functional as F

from models import *
import dataset
from practise import (
    Practise_one_block,
    Practise_all_blocks,
    # metric,
    # train,
    # insert_all_adaptors_for_resnet,
)
from finetune import end_to_end_finetune, validate
# from cam_utils import ClassSpecificImageGeneration

# Prune settings
parser = argparse.ArgumentParser(description="Accelerate networks by PRACTISE")
parser.add_argument(
    "--dataset",
    type=str,
    default="imagenet_fewshot",
    help="training dataset (default: imagenet_fewshot)",
)
parser.add_argument(
    "--eval-dataset",
    type=str,
    default="imagenet",
    help="training dataset (default: imagenet)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--gpu_id", default="7", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
parser.add_argument(
    "--num_sample", type=int, default=50, help="number of samples for training"
)
parser.add_argument(
    "--model", default="resnet34", type=str, help="model name (default: resnet34)"
)
parser.add_argument(
    "--teacher",
    default="",
    type=str,
    metavar="PATH",
    help="path to the pretrained teacher model (default: none)",
)
parser.add_argument(
    "--save",
    default="results",
    type=str,
    metavar="PATH",
    help="path to save pruned model (default: results)",
)
parser.add_argument(
    "--state_dict_path",
    default="",
    type=str,
    metavar="PATH",
    help="path to save pruned model (default: none)",
)
parser.add_argument(
    "--no-pretrained",
    action="store_true",
    default=False,
    help="do not use pretrained weight",
)

parser.add_argument(
    "--rm_blocks", default="", type=str, help="names of removed blocks, split by comma"
)
parser.add_argument(
    "--rm_layers", default="", type=str, help="names of removed blocks, split by comma"
)

parser.add_argument(
    "--practise",
    default="",
    type=str,
    help="blocks for practise",
    choices=["", "one", "all"],
)
parser.add_argument(
    "--FT",
    default="",
    type=str,
    help="method for finetuning",
    choices=["", "BP", "MiR"],
)

parser.add_argument("--opt", default="SGD", type=str, help="opt method (default: SGD)")
parser.add_argument(
    "--lr", type=float, default=0.02, metavar="LR", help="learning rate (default: 0.02)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument("--batch-size", type=int, default=64, help="number of batch size")
parser.add_argument("--epoch", type=int, default=2000, help="number of epoch")
parser.add_argument(
    "--print-freq",
    "-p",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

parser.add_argument("--pmixup", default=False, type=bool)
parser.add_argument("--partmixup", default=0.5, type=float)
parser.add_argument("--beta_a", default=0.3, type=float)
parser.add_argument("--seed", type=int, default=0, help="seed")

parser.add_argument("--group", type=int, default=1, help="group of generated data")
parser.add_argument("--beta", type=float, default=1.0, help="beta")
parser.add_argument("--gamma", type=float, default=0.0, help="gamma")
parser.add_argument("--save_path_head", type=str, default="./", help="save_path_head")
parser.add_argument("--radius", type=float, default=0.05, metavar="radius")
parser.add_argument("--lbns", type=bool, default=False, metavar="lbns")
parser.add_argument("--fft", type=bool, default=False, metavar="fft")


# Custom Dataset for .pickle files
class FakeNetDataset(Dataset):
    def __init__(self, feature_file, label_file):
        self.features = feature_file
        self.labels = label_file

        # Ensure features and labels have matching lengths
        assert len(self.features) == len(self.labels), (
            "Feature and label sizes do not match."
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def main():
    global args
    args = parser.parse_args()
    # python main.py --num_sample 500 --seed 2021 --epoch 100 --practise one --rm_blocks layer1.1 --gpu_id 0
    args.num_sample = 1280
    args.seed = 2021
    args.epoch = 2000  # Training epochs for 3-block pruning
    args.state_dict_path = ""
    args.lr = 0.015  # Slightly lower LR for stability with hybrid method
    args.practise = "all"
    args.rm_blocks = "1"
    args.gpu_id = "0"
    args.dataset = "fakenet"
    args.batch_size = 64
    args.use_msfam = True  # Use Dual-Stage Contrastive-Attention (DSCA)
    args.training_method = 'dsca'  # Two-stage: contrastive then attention
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.save = os.path.join(
        args.save,
        "{}_{}_{}_{}/{}_{}".format(
            args.model, args.dataset, args.practise, args.FT, args.num_sample, args.seed
        ),
    )
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    LOG = logging.getLogger("main")
    time_now = datetime.now()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Use "-" instead of ":"
    logfile = os.path.join(args.save, f"log_{timestamp}.txt")
    FileHandler = logging.FileHandler(logfile, mode="w")
    LOG.addHandler(FileHandler)

    # log print rather than builtin
    # import builtins as __builtin__
    # global builtin_print
    # builtin_print= __builtin__.print
    # __builtin__.print = LOG.info

    print(args)

    if args.eval_dataset == "cifar10":
        args.num_classes = 10
    elif args.eval_dataset == "cifar100":
        args.num_classes = 100
    elif args.eval_dataset == "imagenet":
        args.num_classes = 1000
    elif args.dataset == "fakenet":
        args.num_classes = 1000

    if args.dataset == "imagenet":
        train_loader = dataset.__dict__["imagenet"](True, args.batch_size)
        args.eval_freq = 1
    elif args.dataset == "fakenet":
        for i in range(1, 2):  # 5
            # path = (
            #     "./iterations/iter_2000/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group"
            #     + str(i)
            #     + ".pickle"
            # )
            # path = "image_from_pruned_model-batch/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
            path = (
                "./resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
            )
            # path = "usable_synthetics/new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
            # path = "./psaq/synthetic_images_" + str(i) + ".pickle"
            # path = "./past_src/synthetics/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
            tmp_data = None
            with open(path, "rb") as fp:
                gaussian_data = pickle.load(fp)
                # Remove grayscale conversion, keep RGB data as is
                if (
                    isinstance(gaussian_data, np.ndarray)
                    and gaussian_data.ndim == 4
                    and gaussian_data.shape[1] == 3
                ):
                    # Keep RGB data as is
                    pass
                # gaussian_data = [gaussian_data]
                # print(gaussian_data.shape)
                # print(type(gaussian_data[0]))
                # print(len(gaussian_data))
            if tmp_data is None:
                tmp_data = np.concatenate(gaussian_data, axis=0)
            else:
                tmp_data = np.concatenate(
                    (tmp_data, np.concatenate(gaussian_data, axis=0))
                )

            # path = (
            #     "./iterations/iter_2000/resnet34_labels_hardsample_beta0.1_gamma0.5_group"
            #     + str(i)
            #     + ".pickle"
            # )

            # path = "usable_synthetics/new/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
            # path = "image_from_pruned_model-batch/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
            path = "./resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
            # path = "./psaq/synthetic_labels_" + str(i) + ".pickle"
            # path = "./past_src/synthetics/resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
            tmp_label = None
            with open(path, "rb") as fp:
                labels_list = pickle.load(fp)
                # print(labels_list.shape)
                # print(type(labels_list))
                # labels_list = [labels_list]
            if tmp_label is None:
                tmp_label = np.concatenate(labels_list, axis=0)
            else:
                tmp_label = np.concatenate(
                    (tmp_label, np.concatenate(labels_list, axis=0))
                )

        # Create train_loader and test_loader
        train_dataset = FakeNetDataset(tmp_data, tmp_label)
        # print(tmp_data.shape)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        # Optionally, split data into train and validation/test sets
        val_loader = DataLoader(
            train_dataset, batch_size=args.test_batch_size, shuffle=False
        )

        args.eval_dataset = "imagenet"
        args.dataset = "imagenet_fewshot"
        if args.practise:
            metric_loader = dataset.__dict__["imagenet"](False, args.batch_size)

    else:
        args.eval_freq = 500
        assert args.seed > 0, "Please set seed"
        train_loader = dataset.__dict__[args.dataset](args.num_sample, seed=args.seed)
        if args.practise:
            metric_loader = dataset.__dict__["imagenet"](False, args.batch_size)
        try:
            train_loader.dataset.samples_to_file(os.path.join(args.save, "samples.txt"))
        except:
            print("Not save samples.txt")

    test_loader = dataset.__dict__[args.eval_dataset](False, args.test_batch_size)

    origin_model, all_blocks, origin_lat = build_teacher(
        args.model, args.num_classes, teacher=args.teacher, cuda=args.cuda
    )
    if args.rm_blocks:
        rm_blocks = args.rm_blocks.split(",")
    else:
        rm_blocks = []

    if args.rm_layers:
        rm_layers = args.rm_layers.split(",")
    else:
        rm_layers = []

    if args.practise == "one" and rm_layers == []:
        assert len(rm_blocks) == 1
        pruned_model, _ = Practise_one_block(
            rm_blocks[0], origin_model, origin_lat, train_loader, metric_loader, args
        )
    elif args.practise == "all":
        rm_blocks = []
        # pruned_model, rm_blocks, pruned_lat = Practise_all_blocks(all_blocks, origin_model, origin_lat, train_loader, metric_loader, args, rm_blocks)

        # rm_blocks = ['layer2.2', 'layer3.2', 'layer1.2', 'layer3.3', 'layer1.1']
        # ['layer1.2', 'layer2.2', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.1']
        # rm_blocks = ['layer1.2', 'layer2.3', 'layer2.2']#, 'layer2.3', 'layer3.3']#,'layer3.4']
        # rm_blocks = ['layer4.2', 'layer3.2', 'layer1.2', 'layer1.1', 'layer2.3']
        # rm_blocks = ["layer1.1", "layer1.2", "layer2.2", "layer2.3", "layer3.3"]
        rm_blocks = ["layer1.1", "layer1.2", "layer3.3"]
        pruned_model, _ = Practise_one_block(
            rm_blocks,
            origin_model,
            origin_lat,
            train_loader,
            metric_loader,
            args,
            len(rm_blocks),
        )
        # rm_blocks = ["layer3.3"]  # , 'layer2.1']
        # rm_blocks = ['layer2.1']#, 'layer2.2', 'layer3.2', 'layer3.3'] # 'layer1.2', 'layer2.2',
        # pruned_model1, _ = Practise_one_block(
        #     rm_blocks,
        #     pruned_model,
        #     origin_lat,
        #     train_loader,
        #     metric_loader,
        #     args,
        #     len(rm_blocks),
        # )

    else:
        pruned_model, _, pruned_lat = build_student(
            args.model,
            rm_blocks,
            args.num_classes,
            state_dict_path=args.state_dict_path,
            teacher=args.teacher,
            cuda=args.cuda,
        )
        lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
        print(f"=> latency reduction: {lat_reduction:.2f}%")

    if args.FT:
        validate(test_loader, pruned_model)
        print("=> finetune:")
        # end_to_end_finetune(train_loader, test_loader, pruned_model, origin_model, args)

        # save_path = 'check_point_{:%Y-%m-%d_%H:%M:%S}.tar'.format(time_now)
        save_path = "check_point_pruned.tar"
        save_path = os.path.join(args.save, save_path)
        check_point = {
            "state_dict": pruned_model.state_dict(),
            "rm_blocks": rm_blocks,
        }
        torch.save(check_point, save_path)

        checkpoint = {
            "model_name": "resnet34",
            "rm_blocks": rm_blocks,
            "state_dict": pruned_model.state_dict(),
        }

        # 3. Save the checkpoint to a file
        save_path = "pruned_resnet34_checkpoint.pth"
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    main()