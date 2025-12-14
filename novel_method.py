import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DSCATrainer:
    """
    Dual-Stage Contrastive-Attention Trainer
    Stage 1: Contrastive learning for robust feature alignment
    Stage 2: Attention transfer for fine-grained spatial matching
    """
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.rm_blocks = rm_blocks if rm_blocks else []

        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.teacher.eval()

    def contrastive_loss(self, student_feat, teacher_feat, temperature=0.07):
        """Contrastive loss for instance-level alignment"""
        batch_size = student_feat.size(0)

        # Flatten and normalize
        s_flat = F.normalize(student_feat.view(batch_size, -1), dim=1)
        t_flat = F.normalize(teacher_feat.view(batch_size, -1), dim=1)

        # Similarity matrix
        logits = torch.mm(s_flat, t_flat.t()) / temperature
        labels = torch.arange(batch_size).to(self.device)

        return F.cross_entropy(logits, labels)

    def attention_transfer_loss(self, student_feat, teacher_feat):
        """Spatial attention transfer"""
        # Spatial attention maps
        s_attention = torch.sum(student_feat.pow(2), dim=1, keepdim=True)
        t_attention = torch.sum(teacher_feat.pow(2), dim=1, keepdim=True)

        # Normalize
        bs = student_feat.size(0)
        s_att_norm = F.normalize(s_attention.view(bs, -1), p=2, dim=1)
        t_att_norm = F.normalize(t_attention.view(bs, -1), p=2, dim=1)

        return F.mse_loss(s_att_norm, t_att_norm)

    def get_stage_weights(self, epoch, total_epochs):
        """
        Two-stage strategy:
        Stage 1 (0-40%): Contrastive-focused (coarse alignment)
        Stage 2 (40-100%): Attention-focused (fine-grained refinement)
        """
        progress = epoch / total_epochs

        # Stage 1: Contrastive dominant (0-40%)
        if progress < 0.4:
            lambda_ce = 0.2
            lambda_kd = 0.4
            lambda_contrast = 0.4
            lambda_attention = 0.0
            stage = 1
        # Transition (40-50%)
        elif progress < 0.5:
            lambda_ce = 0.2
            lambda_kd = 0.4
            lambda_contrast = 0.2
            lambda_attention = 0.2
            stage = "transition"
        # Stage 2: Attention dominant (50-100%)
        else:
            lambda_ce = 0.25
            lambda_kd = 0.4
            lambda_contrast = 0.0
            lambda_attention = 0.35
            stage = 2

        return lambda_ce, lambda_kd, lambda_contrast, lambda_attention, stage

    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Training step with dual-stage strategy"""
        self.student.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Set feature extraction
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'

        # Forward pass
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)

        student_logits, student_feat = self.student(images)

        # Get stage-based weights
        lambda_ce, lambda_kd, lambda_contrast, lambda_attention, stage = \
            self.get_stage_weights(epoch, total_epochs)

        # 1. Classification loss
        ce_loss = self.ce_loss(student_logits, labels)

        # 2. Knowledge Distillation loss
        temperature = 4.0
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # 3. Stage-specific losses
        contrast_loss = torch.tensor(0.0).to(self.device)
        attention_loss = torch.tensor(0.0).to(self.device)

        if lambda_contrast > 0:
            contrast_loss = self.contrastive_loss(student_feat, teacher_feat, temperature=0.07)

        if lambda_attention > 0:
            attention_loss = self.attention_transfer_loss(student_feat, teacher_feat)

        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_contrast * contrast_loss +
            lambda_attention * attention_loss
        )

        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'attention_loss': attention_loss.item(),
            'stage': stage,
        }

    def cleanup(self):
        """Cleanup resources"""
        torch.cuda.empty_cache()


# Backward compatibility
ProgressiveBlockRecoveryTrainer = DSCATrainer
SimplifiedMSFAMTrainer = DSCATrainer
EnhancedMSFAMTrainer = DSCATrainer
MSFAMTrainer = DSCATrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000, rm_blocks=None):
    """Train student model using DSCA (Dual-Stage Contrastive-Attention) method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = DSCATrainer(
        student_model, teacher_model,
        rm_blocks=rm_blocks, num_classes=num_classes, device=device
    )

    # Optimizer with proper hyperparameters
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

    # Gradient accumulation for stability
    accumulation_steps = 2

    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'ce_loss': 0,
            'kd_loss': 0,
            'contrast_loss': 0,
            'attention_loss': 0,
        }

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale

            losses = trainer.train_step(images, labels, optimizer, epoch, epochs)

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]

            # Step optimizer with gradient accumulation
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
            print(f"\nEpoch {epoch+1}/{epochs} - Stage: {losses.get('stage', 'N/A')}")
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f}")
            print(f"  KD: {epoch_losses['kd_loss']:.4f} | Contrast: {epoch_losses['contrast_loss']:.4f}")
            print(f"  Attention: {epoch_losses['attention_loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam