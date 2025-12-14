
import os
import argparse
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader

from models import build_teacher
from models.resnet import resnet34_rm_blocks
from dataset import imagenet
from practise import metric, AverageMeter, accuracy


class FakeNetDataset(torch.utils.data.Dataset):
    def __init__(self, feature_file, label_file):
        self.features = feature_file
        self.labels = label_file
        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def main():
    parser = argparse.ArgumentParser(description="Evaluate Data-Free Pruned Model")
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--checkpoint", type=str, default="data_free_pruned_model.pth",
                        help="Path to the pruned model checkpoint")
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data for evaluation instead of ImageNet")
    parser.add_argument("--synthetic_data", type=str, 
                        default="./resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle")
    parser.add_argument("--synthetic_labels", type=str,
                        default="./resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print("=" * 80)
    print("Evaluating Data-Free Pruned Model")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\n=> Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    rm_blocks = checkpoint['rm_blocks']
    
    print(f"=> Removed blocks: {rm_blocks}")
    
    # Build teacher model
    print("\n=> Building teacher model")
    teacher_model, _, _ = build_teacher(
        args.model, num_classes=1000, 
        teacher='', cuda=True
    )
    teacher_model.eval()
    
    # Build student model with same architecture
    print("=> Building pruned student model")
    student_model = resnet34_rm_blocks(
        rm_blocks, pretrained=False, num_classes=1000
    )
    
    # Load trained weights
    student_model.load_state_dict(checkpoint['state_dict'])
    student_model.cuda()
    student_model.eval()
    
    print("=> Model loaded successfully")
    
    # Prepare evaluation data
    if args.use_synthetic:
        print(f"\n=> Loading synthetic data from {args.synthetic_data}")
        with open(args.synthetic_data, 'rb') as f:
            tmp_data = pickle.load(f)
            tmp_data = np.concatenate(tmp_data, axis=0)
        
        with open(args.synthetic_labels, 'rb') as f:
            tmp_label = pickle.load(f)
            tmp_label = np.concatenate(tmp_label, axis=0)
        
        print(f"=> Loaded {len(tmp_data)} synthetic samples")
        
        dataset = FakeNetDataset(tmp_data, tmp_label)
        metric_loader = DataLoader(
            dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=4
        )
    else:
        print("\n=> Using ImageNet validation set")
        metric_loader = imagenet(train=False, batch_size=args.batch_size)
    
    # Run metric evaluation
    print("\n=> Running metric evaluation")
    print("-" * 80)
    
    loss, pruned_acc, teacher_acc, problematic_classes = metric(
        metric_loader, student_model, teacher_model, trained=True
    )
    
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Average Loss: {loss:.4f}")
    print(f"Pruned Model Accuracy: {pruned_acc:.2f}%")
    print(f"Teacher Model Accuracy: {teacher_acc:.2f}%")
    print(f"Accuracy Gap: {teacher_acc - pruned_acc:.2f}%")
    
    if problematic_classes:
        print(f"\nProblematic Classes: {problematic_classes}")
    
    print("=" * 80)
    
    # Save evaluation results
    results = {
        "checkpoint": args.checkpoint,
        "rm_blocks": rm_blocks,
        "loss": loss,
        "pruned_accuracy": pruned_acc,
        "teacher_accuracy": teacher_acc,
        "accuracy_gap": teacher_acc - pruned_acc,
        "problematic_classes": problematic_classes
    }
    
    results_file = args.checkpoint.replace(".pth", "_evaluation_results.pth")
    torch.save(results, results_file)
    print(f"\n=> Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()
