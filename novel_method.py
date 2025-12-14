
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAligner(nn.Module):
    """Learnable channel alignment for feature matching"""
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.align = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
        nn.init.kaiming_normal_(self.align.weight)
        
    def forward(self, x):
        return self.align(x)


class EnhancedMSFAMTrainer:
    """Enhanced MSFAM trainer with progressive learning and adaptive weighting"""
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.rm_blocks = rm_blocks if rm_blocks else []

        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()

        self.teacher.eval()
        
        # Channel aligners (if needed)
        self.channel_aligners = {}
        
        # Track training statistics
        self.epoch_stats = {'easy_samples': 0, 'hard_samples': 0}

    def get_sample_difficulty(self, teacher_logits):
        """Estimate sample difficulty based on teacher confidence"""
        probs = F.softmax(teacher_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        # Lower confidence = harder sample
        difficulty = 1.0 - max_probs
        return difficulty

    def curriculum_weight(self, difficulty, epoch, total_epochs):
        """Progressive curriculum: start with easy samples, gradually add harder ones"""
        progress = epoch / total_epochs
        # Threshold increases from 0.3 to 1.0
        threshold = 0.3 + 0.7 * progress
        
        # Samples below threshold get full weight, above get reduced weight
        weights = torch.where(
            difficulty < threshold,
            torch.ones_like(difficulty),
            torch.exp(-3 * (difficulty - threshold))
        )
        return weights

    def adaptive_loss_weights(self, epoch, total_epochs):
        """Adaptive loss weights based on training progress"""
        progress = epoch / total_epochs
        
        if progress < 0.2:
            # Early: focus on feature matching
            lambda_ce = 0.1
            lambda_kd = 0.3
            lambda_feat = 0.4
            lambda_channel = 0.2
        elif progress < 0.5:
            # Mid-early: balanced
            lambda_ce = 0.15
            lambda_kd = 0.35
            lambda_feat = 0.35
            lambda_channel = 0.15
        elif progress < 0.8:
            # Mid-late: emphasize KD
            lambda_ce = 0.2
            lambda_kd = 0.45
            lambda_feat = 0.25
            lambda_channel = 0.1
        else:
            # Late: focus on classification and KD
            lambda_ce = 0.25
            lambda_kd = 0.5
            lambda_feat = 0.2
            lambda_channel = 0.05
            
        return lambda_ce, lambda_kd, lambda_feat, lambda_channel

    def multi_scale_feature_loss(self, student_feat, teacher_feat):
        """Multi-scale feature alignment with spatial attention"""
        bs, c, h, w = student_feat.shape
        
        # 1. Global feature alignment
        global_loss = self.mse_loss(student_feat, teacher_feat)
        
        # 2. Spatial attention alignment
        s_spatial = student_feat.pow(2).mean(1, keepdim=True)  # [B, 1, H, W]
        t_spatial = teacher_feat.pow(2).mean(1, keepdim=True)
        
        s_spatial_norm = F.normalize(s_spatial.view(bs, -1), dim=1)
        t_spatial_norm = F.normalize(t_spatial.view(bs, -1), dim=1)
        spatial_loss = self.mse_loss(s_spatial_norm, t_spatial_norm)
        
        # 3. Channel-wise correlation
        s_channel = student_feat.view(bs, c, -1).mean(2)  # [B, C]
        t_channel = teacher_feat.view(bs, c, -1).mean(2)
        
        s_channel_norm = F.normalize(s_channel, dim=1)
        t_channel_norm = F.normalize(t_channel, dim=1)
        channel_loss = self.mse_loss(s_channel_norm, t_channel_norm)
        
        return global_loss + 0.5 * spatial_loss + 0.5 * channel_loss

    def instance_level_loss(self, student_feat, teacher_feat):
        """Instance-level normalized MSE loss"""
        bs = student_feat.size(0)
        s_feat_flat = student_feat.view(bs, -1)
        t_feat_flat = teacher_feat.view(bs, -1)
        
        s_norm = s_feat_flat / (s_feat_flat.norm(dim=1, keepdim=True) + 1e-8)
        t_norm = t_feat_flat / (t_feat_flat.norm(dim=1, keepdim=True) + 1e-8)
        
        return torch.mean((s_norm - t_norm) ** 2)

    def class_level_loss(self, student_feat, teacher_feat):
        """Class-level feature distribution matching"""
        bs = student_feat.size(0)
        s_feat_flat = student_feat.view(bs, -1)
        t_feat_flat = teacher_feat.view(bs, -1)
        
        # Transpose to get [C, B] and apply normalization
        s_class = s_feat_flat.t()
        t_class = t_feat_flat.t()
        
        s_class_norm = s_class / (s_class.norm(dim=1, keepdim=True) + 1e-8)
        t_class_norm = t_class / (t_class.norm(dim=1, keepdim=True) + 1e-8)
        
        return torch.mean((s_class_norm - t_class_norm) ** 2)

    def train_step(self, images, labels, optimizer, epoch, total_epochs, accumulation_steps=2):
        """Enhanced training step with all improvements"""
        self.student.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'

        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)
            
            # Estimate sample difficulty
            difficulty = self.get_sample_difficulty(teacher_logits)
            curriculum_weights = self.curriculum_weight(difficulty, epoch, total_epochs)

        # Get student outputs
        student_logits, student_feat = self.student(images)

        # Get adaptive loss weights
        lambda_ce, lambda_kd, lambda_feat, lambda_channel = self.adaptive_loss_weights(epoch, total_epochs)

        # 1. Classification loss (weighted by curriculum)
        if len(labels.shape) == 1:
            ce_loss_per_sample = F.cross_entropy(student_logits, labels, reduction='none', label_smoothing=0.1)
            ce_loss = (ce_loss_per_sample * curriculum_weights).mean()
        else:
            ce_loss_per_sample = -torch.sum(labels * F.log_softmax(student_logits, dim=1), dim=1)
            ce_loss = (ce_loss_per_sample * curriculum_weights).mean()

        # 2. KD loss with adaptive temperature
        base_temp = 4.0
        progress = epoch / total_epochs
        temperature = base_temp * (1.0 - 0.4 * progress)  # Gradually decrease temperature

        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        
        kd_loss_per_sample = F.kl_div(soft_student, soft_teacher, reduction='none').sum(dim=1)
        kd_loss = (kd_loss_per_sample * curriculum_weights).mean() * (temperature ** 2)

        # 3. Multi-scale feature matching loss
        feat_loss = self.multi_scale_feature_loss(student_feat, teacher_feat)

        # 4. Instance and class level losses
        inst_loss = self.instance_level_loss(student_feat, teacher_feat)
        class_loss = self.class_level_loss(student_feat, teacher_feat)
        
        channel_loss = inst_loss + class_loss

        # Combined loss with gradient accumulation scaling
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * feat_loss +
            lambda_channel * channel_loss
        ) / accumulation_steps

        # Backward
        total_loss.backward()

        # Return scaled losses for logging
        return {
            'total_loss': total_loss.item() * accumulation_steps,
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'feat_loss': feat_loss.item(),
            'channel_loss': channel_loss.item(),
            'avg_difficulty': difficulty.mean().item(),
            'temperature': temperature
        }

    def cleanup(self):
        """Cleanup resources"""
        self.channel_aligners.clear()
        torch.cuda.empty_cache()


# Backward compatibility
SimplifiedMSFAMTrainer = EnhancedMSFAMTrainer
MSFAMTrainer = EnhancedMSFAMTrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000, rm_blocks=None):
    """Train student model using Enhanced MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = EnhancedMSFAMTrainer(
        student_model, teacher_model, 
        rm_blocks=rm_blocks, num_classes=num_classes, device=device
    )

    # Optimizer with Nesterov momentum
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, student_model.parameters()),
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

    # Gradient accumulation
    accumulation_steps = 2

    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'ce_loss': 0,
            'kd_loss': 0,
            'feat_loss': 0,
            'channel_loss': 0,
            'avg_difficulty': 0,
        }

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale

            losses = trainer.train_step(
                images, labels, optimizer, epoch, epochs, accumulation_steps
            )

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]

            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, student_model.parameters()), 
                    max_norm=5.0
                )
                optimizer.step()
                optimizer.zero_grad()

            # Clear cache periodically
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        # Calculate epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        # Print epoch statistics
        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f} | "
                  f"KD: {epoch_losses['kd_loss']:.4f} | Feature: {epoch_losses['feat_loss']:.4f}")
            print(f"  Channel: {epoch_losses['channel_loss']:.4f} | Difficulty: {epoch_losses['avg_difficulty']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam
