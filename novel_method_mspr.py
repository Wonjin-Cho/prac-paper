
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class FeatureDiscriminator(nn.Module):
    """Discriminator to distinguish student vs teacher features"""
    def __init__(self, feature_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class SampleDifficultyEstimator:
    """Estimate sample difficulty based on teacher confidence"""
    def __init__(self):
        self.difficulties = []
    
    def estimate(self, logits):
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        # Lower confidence = harder sample
        difficulty = 1.0 - max_probs
        return difficulty
    
    def get_curriculum_indices(self, all_logits, current_epoch, total_epochs):
        """Return indices of samples to use based on curriculum"""
        difficulties = self.estimate(all_logits)
        
        # Progressive difficulty: start with easy samples, gradually add harder ones
        progress = current_epoch / total_epochs
        threshold = 0.3 + 0.7 * progress  # Start at 30%, end at 100%
        
        # Sort by difficulty and select easiest threshold%
        sorted_indices = torch.argsort(difficulties)
        num_samples = int(len(sorted_indices) * threshold)
        
        return sorted_indices[:num_samples]


class MSPRTrainer:
    """Multi-Stage Progressive Recovery Trainer"""
    def __init__(self, student_model, teacher_model, rm_blocks, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.rm_blocks = rm_blocks
        self.device = device
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            self.student.get_feat = 'pre_GAP'
            _, student_feat = self.student(dummy_input)
            feat_dim = student_feat.view(1, -1).size(1)
        
        # Initialize discriminator
        self.discriminator = FeatureDiscriminator(feat_dim).to(device)
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Difficulty estimator
        self.difficulty_estimator = SampleDifficultyEstimator()
        
        # EMA model for self-distillation
        self.ema_model = self._create_ema_model()
        self.ema_decay = 0.999
        
        self.teacher.eval()
        
        # Track which blocks are unfrozen
        self.unfrozen_blocks = set()
    
    def _create_ema_model(self):
        """Create EMA model for self-distillation"""
        import copy
        ema_model = copy.deepcopy(self.student)
        for param in ema_model.parameters():
            param.requires_grad = False
        ema_model.eval()
        return ema_model
    
    def _update_ema(self):
        """Update EMA model with current student weights"""
        with torch.no_grad():
            for ema_param, student_param in zip(self.ema_model.parameters(), self.student.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
    
    def _progressive_unfreeze(self, epoch, total_epochs):
        """Progressively unfreeze layers from deep to shallow"""
        # Parse block names to get layer and block numbers
        block_info = []
        for block_name in self.rm_blocks:
            parts = block_name.split('.')
            if len(parts) >= 2:
                layer_num = int(parts[0].replace('layer', ''))
                block_num = int(parts[1])
                block_info.append((layer_num, block_num, block_name))
        
        # Sort by layer (descending) then block number
        block_info.sort(reverse=True)
        
        # Determine how many blocks to unfreeze based on epoch
        num_stages = len(block_info)
        blocks_per_stage = total_epochs // (num_stages + 1)
        current_stage = min(epoch // blocks_per_stage, num_stages)
        
        # Unfreeze blocks up to current stage
        for i in range(current_stage):
            if i < len(block_info):
                block_name = block_info[i][2]
                if block_name not in self.unfrozen_blocks:
                    self._unfreeze_block_adaptors(block_name)
                    self.unfrozen_blocks.add(block_name)
                    print(f"Unfroze adaptors for block: {block_name}")
    
    def _unfreeze_block_adaptors(self, block_name):
        """Unfreeze adaptors related to a specific block"""
        # Find and unfreeze all adaptors related to this block
        for name, param in self.student.named_parameters():
            if 'adaptor' in name.lower() and block_name in name:
                param.requires_grad = True
    
    def train_step(self, images, labels, optimizer, disc_optimizer, epoch, total_epochs, use_augmentation=True):
        self.student.train()
        self.discriminator.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Apply progressive unfreezing
        self._progressive_unfreeze(epoch, total_epochs)
        
        # Data augmentation for robustness
        if use_augmentation and torch.rand(1).item() > 0.3:
            # Random augmentation
            if torch.rand(1).item() > 0.5:
                # Mixup
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(images.size(0)).to(self.device)
                images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                mixed = True
            else:
                # Cutout
                mask = torch.ones_like(images)
                h, w = images.size(2), images.size(3)
                cut_size = min(h, w) // 4
                cy, cx = np.random.randint(0, h), np.random.randint(0, w)
                y1 = np.clip(cy - cut_size // 2, 0, h)
                y2 = np.clip(cy + cut_size // 2, 0, h)
                x1 = np.clip(cx - cut_size // 2, 0, w)
                x2 = np.clip(cx + cut_size // 2, 0, w)
                mask[:, :, y1:y2, x1:x2] = 0
                images = images * mask
                mixed = False
        else:
            mixed = False
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        self.ema_model.get_feat = 'pre_GAP'
        
        # Forward pass
        student_logits, student_features = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(images)
            ema_logits, ema_features = self.ema_model(images)
        
        # Flatten features
        s_feat_flat = student_features.view(student_features.size(0), -1)
        t_feat_flat = teacher_features.view(teacher_features.size(0), -1)
        ema_feat_flat = ema_features.view(ema_features.size(0), -1)
        
        # ===== STUDENT TRAINING =====
        optimizer.zero_grad()
        
        # 1. Knowledge Distillation Loss (from teacher)
        temperature = 4.0
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 2. Self-Distillation Loss (from EMA model)
        self_kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(ema_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 3. Classification Loss
        if mixed:
            ce_loss = lam * self.ce_loss(student_logits, labels_a) + (1 - lam) * self.ce_loss(student_logits, labels_b)
        else:
            ce_loss = self.ce_loss(student_logits, labels)
        
        # 4. Feature Matching Loss
        feat_loss = F.mse_loss(s_feat_flat, t_feat_flat)
        
        # 5. Adversarial Loss (fool discriminator)
        disc_pred_student = self.discriminator(s_feat_flat)
        # Want student features to be classified as teacher (label=1)
        adv_loss = self.bce_loss(disc_pred_student, torch.ones_like(disc_pred_student))
        
        # 6. Feature diversity loss (prevent mode collapse)
        # Encourage different samples to have different features
        feat_cov = torch.mm(s_feat_flat.t(), s_feat_flat) / s_feat_flat.size(0)
        diversity_loss = -torch.mean(torch.abs(feat_cov))
        
        # Adaptive weights based on training progress
        progress = epoch / total_epochs
        w_kd = 0.3
        w_self_kd = 0.1 + 0.2 * progress  # Increase self-distillation weight over time
        w_ce = 0.2
        w_feat = 0.2 - 0.1 * progress  # Decrease direct feature matching over time
        w_adv = 0.1 + 0.1 * progress  # Increase adversarial weight over time
        w_div = 0.05
        
        total_student_loss = (
            w_kd * kd_loss +
            w_self_kd * self_kd_loss +
            w_ce * ce_loss +
            w_feat * feat_loss +
            w_adv * adv_loss +
            w_div * diversity_loss
        )
        
        total_student_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Update EMA model
        self._update_ema()
        
        # ===== DISCRIMINATOR TRAINING =====
        disc_optimizer.zero_grad()
        
        # Detach to not backprop through student
        disc_pred_student = self.discriminator(s_feat_flat.detach())
        disc_pred_teacher = self.discriminator(t_feat_flat.detach())
        
        # Real (teacher) = 1, Fake (student) = 0
        disc_loss = (
            self.bce_loss(disc_pred_teacher, torch.ones_like(disc_pred_teacher)) +
            self.bce_loss(disc_pred_student, torch.zeros_like(disc_pred_student))
        ) / 2
        
        disc_loss.backward()
        disc_optimizer.step()
        
        return {
            'total_loss': total_student_loss.item(),
            'kd_loss': kd_loss.item(),
            'self_kd_loss': self_kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'feat_loss': feat_loss.item(),
            'adv_loss': adv_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'disc_loss': disc_loss.item()
        }


def train_with_mspr(student_model, teacher_model, train_loader, rm_blocks, epochs=100, lr=0.01):
    """Train using Multi-Stage Progressive Recovery"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = MSPRTrainer(student_model, teacher_model, rm_blocks, device)
    
    # Separate optimizers for student and discriminator
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    disc_optimizer = torch.optim.Adam(
        trainer.discriminator.parameters(),
        lr=0.001,
        betas=(0.5, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        epoch_losses = defaultdict(float)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            losses = trainer.train_step(
                images, labels, optimizer, disc_optimizer, 
                epoch, epochs, use_augmentation=True
            )
            
            for key, value in losses.items():
                epoch_losses[key] += value
        
        scheduler.step()
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
                print(f"  {key}: {epoch_losses[key]:.4f}", end="")
            print()
    
    return student_model
