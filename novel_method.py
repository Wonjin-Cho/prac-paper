import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimplifiedMSFAMTrainer:
    """Memory-efficient MSFAM trainer without hooks"""
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes

        # Standard losses only
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()

        self.teacher.eval()

    def nmse_loss(self, p, z):
        """Normalized MSE loss"""
        p_norm = p / (p.norm(dim=1, keepdim=True) + 1e-8)
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return torch.mean((p_norm - z_norm) ** 2)

    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Memory-efficient training step"""
        self.student.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'

        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)

        # Get student outputs
        student_logits, student_feat = self.student(images)

        # 1. Classification loss
        if len(labels.shape) == 1:
            ce_loss = self.ce_loss(student_logits, labels)
        else:
            ce_loss = -torch.mean(torch.sum(labels * F.log_softmax(student_logits, dim=1), dim=1))

        # 2. KD loss with adaptive temperature
        base_temp = 4.0
        progress = epoch / total_epochs
        temperature = base_temp * (1.0 - 0.5 * progress)

        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # 3. Feature matching loss (simple MSE)
        feat_loss = self.mse_loss(student_feat, teacher_feat)

        # 4. Normalized feature loss
        nmse_feat_loss = self.nmse_loss(student_feat, teacher_feat)

        # Progressive loss weights
        if epoch < 200:
            lambda_ce = 0.3
            lambda_kd = 0.5
            lambda_feat = 0.1
            lambda_nmse = 0.1
        elif epoch < 1000:
            lambda_ce = 0.2
            lambda_kd = 0.5
            lambda_feat = 0.15
            lambda_nmse = 0.15
        else:
            lambda_ce = 0.15
            lambda_kd = 0.45
            lambda_feat = 0.2
            lambda_nmse = 0.2

        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * feat_loss +
            lambda_nmse * nmse_feat_loss
        )

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'feat_loss': feat_loss.item() + nmse_feat_loss.item()
        }

    def cleanup(self):
        """No cleanup needed in simplified version"""
        pass


# Backward compatibility
MSFAMTrainer = SimplifiedMSFAMTrainer
EnhancedMSFAMTrainer = SimplifiedMSFAMTrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000):
    """Train student model using MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = SimplifiedMSFAMTrainer(student_model, teacher_model, num_classes=num_classes, device=device)

    # Optimizer with weight decay
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    # Cosine annealing with warmup
    warmup_epochs = int(0.1 * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )

    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'ce_loss': 0,
            'kd_loss': 0,
            'feat_loss': 0
        }

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale

            losses = trainer.train_step(images, labels, optimizer, epoch, epochs)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

        if epoch >= warmup_epochs:
            scheduler.step()

        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f} | "
                  f"KD: {epoch_losses['kd_loss']:.4f} | Feature: {epoch_losses['feat_loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam