
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MomentumEncoder:
    """Momentum-updated teacher encoder for stable contrastive targets"""
    def __init__(self, teacher_model, momentum=0.996):
        self.momentum = momentum
        self.teacher = teacher_model
        
    @torch.no_grad()
    def update(self, student_model):
        """Update momentum encoder with student parameters"""
        for teacher_param, student_param in zip(self.teacher.parameters(), student_model.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1 - self.momentum) * student_param.data


class HardNegativeContrastiveLoss(nn.Module):
    """Contrastive loss with hard negative mining for few-shot learning"""
    def __init__(self, temperature=0.07, hard_negative_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, student_feat, teacher_feat):
        batch_size = student_feat.size(0)
        
        # Flatten and normalize
        student_feat = F.normalize(student_feat.view(batch_size, -1), dim=1)
        teacher_feat = F.normalize(teacher_feat.view(batch_size, -1), dim=1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        logits = torch.mm(student_feat, teacher_feat.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(student_feat.device)
        
        # Hard negative mining: identify hardest negatives per sample
        with torch.no_grad():
            # Mask out positive pairs (diagonal)
            mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            negative_logits = logits.masked_fill(mask, float('-inf'))
            
            # Find hardest negatives (highest similarity among negatives)
            hard_negatives = negative_logits.max(dim=1)[0]
        
        # Standard contrastive loss
        base_loss = F.cross_entropy(logits, labels)
        
        # Hard negative loss: additional penalty for hard negatives
        # We want to push hard negatives further away
        positive_logits = logits[torch.arange(batch_size), labels]
        hard_negative_loss = F.relu(hard_negatives - positive_logits + 0.5).mean()
        
        total_loss = base_loss + self.hard_negative_weight * hard_negative_loss
        
        return total_loss, base_loss, hard_negative_loss


class EnhancedContrastiveTrainer:
    """Enhanced Contrastive Knowledge Distillation with Momentum Encoder and Hard Negative Mining"""
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.rm_blocks = rm_blocks if rm_blocks else []
        
        # Create momentum encoder
        self.momentum_encoder = MomentumEncoder(self.teacher, momentum=0.996)
        
        # Loss functions
        self.contrastive_loss_fn = HardNegativeContrastiveLoss(
            temperature=0.07,
            hard_negative_weight=1.5
        )
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.teacher.eval()
    
    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Training step with enhanced contrastive learning"""
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        # Forward pass
        student_logits, student_feat = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)
        
        # Adaptive loss weights based on training progress
        progress = epoch / total_epochs
        
        # Gradually shift from KD to contrastive learning
        if progress < 0.3:
            # Early stage: focus on KD for basic knowledge transfer
            lambda_ce = 0.25
            lambda_kd = 0.50
            lambda_contrast = 0.25
        elif progress < 0.7:
            # Middle stage: balance between KD and contrastive
            lambda_ce = 0.20
            lambda_kd = 0.40
            lambda_contrast = 0.40
        else:
            # Late stage: focus on contrastive for fine-grained alignment
            lambda_ce = 0.15
            lambda_kd = 0.30
            lambda_contrast = 0.55
        
        # 1. Classification loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 2. Knowledge Distillation loss with adaptive temperature
        temperature = 4.0 * (1.0 - 0.2 * progress)  # Temperature annealing
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 3. Enhanced contrastive loss with hard negative mining
        contrast_loss, base_contrast, hard_neg_loss = self.contrastive_loss_fn(
            student_feat, teacher_feat
        )
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_contrast * contrast_loss
        )
        
        total_loss.backward()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'base_contrast': base_contrast.item(),
            'hard_neg_loss': hard_neg_loss.item(),
        }
    
    def update_momentum_encoder(self):
        """Update momentum encoder after optimizer step"""
        self.momentum_encoder.update(self.student)
    
    def cleanup(self):
        """Cleanup resources"""
        torch.cuda.empty_cache()


# Backward compatibility
DSCATrainer = EnhancedContrastiveTrainer
ProgressiveBlockRecoveryTrainer = EnhancedContrastiveTrainer
SimplifiedMSFAMTrainer = EnhancedContrastiveTrainer
EnhancedMSFAMTrainer = EnhancedContrastiveTrainer
MSFAMTrainer = EnhancedContrastiveTrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000, rm_blocks=None):
    """Train student model using Enhanced Contrastive method with Hard Negative Mining and Momentum Encoder"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = EnhancedContrastiveTrainer(
        student_model, teacher_model,
        rm_blocks=rm_blocks, num_classes=num_classes, device=device
    )
    
    # Optimizer with proper hyperparameters for few-shot setting
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Cosine annealing with warmup
    warmup_epochs = int(0.05 * epochs)  # 5% warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    
    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'ce_loss': 0,
            'kd_loss': 0,
            'contrast_loss': 0,
            'base_contrast': 0,
            'hard_neg_loss': 0,
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, student_model.parameters()),
                max_norm=5.0
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update momentum encoder after each batch
            trainer.update_momentum_encoder()
            
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
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f}")
            print(f"  KD: {epoch_losses['kd_loss']:.4f} | Contrast: {epoch_losses['contrast_loss']:.4f}")
            print(f"  Base Contrast: {epoch_losses['base_contrast']:.4f} | Hard Neg: {epoch_losses['hard_neg_loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam
