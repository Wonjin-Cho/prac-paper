
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

from models import *
from finetune import AverageMeter, validate, accuracy
from compute_flops import compute_MACs_params


class DataFreeBlockPruning:
    """
    Solves conceptual problems in data-free block pruning:
    1. Distribution mismatch via BN recalibration
    2. Knowledge gap via progressive distillation
    3. Channel mismatch via adaptive adapters
    4. BN corruption via running statistics correction
    5. Limited coverage via gradient-based data augmentation
    """
    
    def __init__(self, teacher_model, student_model, synthetic_data, args):
        self.teacher = teacher_model.cuda().eval()
        self.student = student_model.cuda()
        self.synthetic_data = synthetic_data
        self.args = args
        self.device = 'cuda'
        
        # Enable feature extraction
        self.teacher.get_feat = 'pre_GAP'
        self.student.get_feat = 'pre_GAP'
        
    def recalibrate_bn_statistics(self):
        """
        Solution to Problem 4: BN Statistics Corruption
        Based on "Diverse Sample Generation" (ZeroQ)
        """
        print("=> Recalibrating BN statistics with synthetic data")
        self.student.train()
        
        # Reset BN running stats
        for m in self.student.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = 0.1
        
        # Forward pass to populate BN stats
        with torch.no_grad():
            for data, _ in self.synthetic_data:
                data = data.cuda()
                _ = self.student(data)
        
        self.student.eval()
        print("=> BN recalibration complete")
    
    def compute_channel_alignment_loss(self, student_feat, teacher_feat):
        """
        Solution to Problem 3: Channel Dimension Mismatch
        Adaptive channel alignment with attention
        """
        B, C_s, H, W = student_feat.shape
        _, C_t, _, _ = teacher_feat.shape
        
        if C_s != C_t:
            # Use 1x1 conv for channel alignment
            if not hasattr(self, 'channel_adapters'):
                self.channel_adapters = {}
            
            key = f"{C_s}_{C_t}"
            if key not in self.channel_adapters:
                adapter = nn.Conv2d(C_s, C_t, 1, bias=False).cuda()
                nn.init.kaiming_normal_(adapter.weight)
                self.channel_adapters[key] = adapter
            
            student_feat = self.channel_adapters[key](student_feat)
        
        # Spatial attention alignment
        s_spatial = student_feat.pow(2).mean(1, keepdim=True)
        t_spatial = teacher_feat.pow(2).mean(1, keepdim=True)
        
        spatial_loss = F.mse_loss(
            F.normalize(s_spatial.view(B, -1), dim=1),
            F.normalize(t_spatial.view(B, -1), dim=1)
        )
        
        # Feature MSE loss
        feat_loss = F.mse_loss(student_feat, teacher_feat)
        
        return feat_loss + 0.5 * spatial_loss
    
    def progressive_knowledge_distillation(self, epoch, total_epochs):
        """
        Solution to Problem 2: Knowledge Gap Amplification
        Progressive temperature and loss weight scheduling
        """
        # Temperature annealing (high -> low)
        progress = epoch / total_epochs
        temperature = 4.0 * (1.0 - 0.7 * progress)
        
        # Loss weight scheduling
        if progress < 0.3:
            # Early: focus on feature matching
            lambda_ce = 0.1
            lambda_kd = 0.3
            lambda_feat = 0.6
        elif progress < 0.7:
            # Mid: balanced
            lambda_ce = 0.2
            lambda_kd = 0.5
            lambda_feat = 0.3
        else:
            # Late: focus on classification
            lambda_ce = 0.3
            lambda_kd = 0.5
            lambda_feat = 0.2
        
        return temperature, lambda_ce, lambda_kd, lambda_feat
    
    def gradient_based_data_refinement(self, data, target):
        """
        Solution to Problem 5: Limited Feature Coverage
        Refine synthetic data using gradient signals
        Based on "DFAD: Data-Free Adversarial Distillation"
        """
        data = data.clone().detach().requires_grad_(True)
        
        # Get teacher output
        with torch.no_grad():
            t_logits, _ = self.teacher(data)
        
        # Student output
        s_logits, _ = self.student(data)
        
        # Maximize KL divergence to explore diverse regions
        kl_loss = F.kl_div(
            F.log_softmax(s_logits, dim=1),
            F.softmax(t_logits, dim=1),
            reduction='batchmean'
        )
        
        # Get gradient
        grad = torch.autograd.grad(kl_loss, data)[0]
        
        # Perturb data in gradient direction (small step)
        refined_data = data + 0.01 * grad.sign()
        refined_data = torch.clamp(refined_data, 0, 1)
        
        return refined_data.detach()
    
    def multi_scale_feature_alignment(self, images):
        """
        Solution to Problem 1: Distribution Mismatch
        Multi-scale feature distillation
        """
        losses = []
        
        # Multiple resolutions for robustness
        scales = [1.0, 0.875, 1.125]
        
        for scale in scales:
            if scale != 1.0:
                size = int(224 * scale)
                scaled_images = F.interpolate(
                    images, size=(size, size), 
                    mode='bilinear', align_corners=False
                )
            else:
                scaled_images = images
            
            with torch.no_grad():
                t_logits, t_feat = self.teacher(scaled_images)
            
            s_logits, s_feat = self.student(scaled_images)
            
            # Feature alignment loss
            feat_loss = self.compute_channel_alignment_loss(s_feat, t_feat)
            losses.append(feat_loss)
        
        return sum(losses) / len(losses)
    
    def train_epoch(self, optimizer, epoch, total_epochs):
        """Main training loop with all solutions integrated"""
        self.student.train()
        
        # Progressive distillation parameters
        temperature, lambda_ce, lambda_kd, lambda_feat = \
            self.progressive_knowledge_distillation(epoch, total_epochs)
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.synthetic_data):
            data = data.cuda()
            target = target.cuda()
            
            # Apply gradient-based refinement periodically
            if batch_idx % 5 == 0 and epoch > 100:
                data = self.gradient_based_data_refinement(data, target)
            
            # Get teacher predictions
            with torch.no_grad():
                t_logits, t_feat = self.teacher(data)
            
            # Get student predictions
            s_logits, s_feat = self.student(data)
            
            # Loss 1: Classification (CE or soft labels)
            if len(target.shape) > 1:  # Soft labels
                ce_loss = -torch.mean(
                    torch.sum(target * F.log_softmax(s_logits, dim=1), dim=1)
                )
            else:
                ce_loss = F.cross_entropy(s_logits, target, label_smoothing=0.1)
            
            # Loss 2: Knowledge Distillation with temperature
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=1),
                F.softmax(t_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Loss 3: Multi-scale feature alignment
            feat_loss = self.multi_scale_feature_alignment(data)
            
            # Combined loss
            loss = (lambda_ce * ce_loss + 
                   lambda_kd * kd_loss + 
                   lambda_feat * feat_loss)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{total_epochs}, Batch {batch_idx}: "
                      f"Loss={loss.item():.4f}, CE={ce_loss.item():.4f}, "
                      f"KD={kd_loss.item():.4f}, Feat={feat_loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(self, epochs=2000, lr=0.02):
        """Full training pipeline"""
        # Step 1: BN recalibration
        self.recalibrate_bn_statistics()
        
        # Step 2: Setup optimizer with layer-wise LR
        params = []
        for name, param in self.student.named_parameters():
            if 'adapter' in name or 'channel' in name:
                params.append({'params': param, 'lr': lr * 2.0})
            else:
                params.append({'params': param, 'lr': lr})
        
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, 
            weight_decay=1e-4, nesterov=True
        )
        
        # Step 3: Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # Step 4: Training loop
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(optimizer, epoch, epochs)
            scheduler.step()
            
            if epoch % 200 == 0:
                print(f"\n=== Epoch {epoch}/{epochs} ===")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
                
                # Re-calibrate BN periodically
                self.recalibrate_bn_statistics()
        
        # Final BN calibration
        print("\n=> Final BN recalibration")
        self.recalibrate_bn_statistics()
        
        return self.student


class FakeNetDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Data-Free Block Pruning")
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Data-Free Block Pruning with Conceptual Problem Solutions")
    print("=" * 80)
    print("\nSolutions implemented:")
    print("1. BN Statistics Recalibration (Problem: BN corruption)")
    print("2. Progressive Knowledge Distillation (Problem: Knowledge gap)")
    print("3. Adaptive Channel Alignment (Problem: Dimension mismatch)")
    print("4. Gradient-based Data Refinement (Problem: Limited coverage)")
    print("5. Multi-scale Feature Alignment (Problem: Distribution mismatch)")
    print("=" * 80 + "\n")
    
    # Load synthetic data
    print("=> Loading synthetic data")
    data_path = "./resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle"
    label_path = "./resnet34_labels_hardsample_beta0.1_gamma0.5_group1.pickle"
    
    with open(data_path, 'rb') as f:
        tmp_data = pickle.load(f)
        tmp_data = np.concatenate(tmp_data, axis=0)
    
    with open(label_path, 'rb') as f:
        tmp_label = pickle.load(f)
        tmp_label = np.concatenate(tmp_label, axis=0)
    
    print(f"=> Loaded {len(tmp_data)} synthetic images")
    
    # Create data loader
    dataset = FakeNetDataset(tmp_data, tmp_label)
    synthetic_loader = DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4
    )
    
    # Build models
    print("\n=> Building teacher model")
    teacher_model, _, _ = build_teacher(
        args.model, num_classes=1000, 
        teacher='', cuda=True
    )
    
    print("=> Building pruned student model")
    # Prune 3 blocks as example
    rm_blocks = ["layer1.1", "layer1.2", "layer3.3"]
    student_model, _, _ = build_student(
        args.model, rm_blocks, num_classes=1000,
        state_dict_path='', teacher='', cuda=True
    )
    
    print(f"=> Removed blocks: {rm_blocks}")
    
    # Initialize data-free pruning framework
    df_pruning = DataFreeBlockPruning(
        teacher_model, student_model, 
        synthetic_loader, args
    )
    
    # Train with all solutions
    print("\n=> Starting data-free training with problem solutions")
    trained_student = df_pruning.train(
        epochs=args.epoch, lr=args.lr
    )
    
    print("\n=> Training complete!")
    print("=> Model ready for evaluation or deployment")
    
    # Save checkpoint
    checkpoint = {
        "model_name": args.model,
        "rm_blocks": rm_blocks,
        "state_dict": trained_student.state_dict(),
    }
    save_path = "data_free_pruned_model.pth"
    torch.save(checkpoint, save_path)
    print(f"=> Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
