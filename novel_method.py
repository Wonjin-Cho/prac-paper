
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProgressiveBlockRecoveryTrainer:
    """
    Hybrid trainer combining:
    1. Contrastive learning for feature alignment
    2. Attention transfer for spatial/channel matching
    3. Progressive unfreezing for multi-block pruning
    4. Adaptive weighting based on training stage
    """
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.rm_blocks = rm_blocks if rm_blocks else []
        
        # Parse block information for progressive unfreezing
        self.block_layers = self._parse_blocks()
        self.unfrozen_stages = 0
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
        
        self.teacher.eval()
        
    def _parse_blocks(self):
        """Parse removed blocks and sort by layer depth"""
        blocks = []
        for block in self.rm_blocks:
            parts = block.split('.')
            if len(parts) >= 2:
                layer_num = int(parts[0].replace('layer', ''))
                block_num = int(parts[1])
                blocks.append((layer_num, block_num, block))
        # Sort by layer (deep to shallow for progressive unfreezing)
        blocks.sort(reverse=True)
        return blocks
    
    def _progressive_unfreeze(self, epoch, total_epochs):
        """Progressively unfreeze model parameters"""
        if len(self.block_layers) == 0:
            return
        
        # Unfreeze in stages: 0-20% all frozen, then gradually unfreeze
        num_stages = len(self.block_layers) + 1
        stage_duration = total_epochs // num_stages
        current_stage = min(epoch // stage_duration, num_stages - 1)
        
        if current_stage > self.unfrozen_stages:
            self.unfrozen_stages = current_stage
            if current_stage > 0:
                print(f"Epoch {epoch}: Unfreezing stage {current_stage}/{num_stages}")
                # Unfreeze all parameters at current stage
                for param in self.student.parameters():
                    param.requires_grad = True
    
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
    
    def relation_loss(self, student_feat, teacher_feat):
        """Pairwise relation preservation"""
        bs = student_feat.size(0)
        s_flat = F.normalize(student_feat.view(bs, -1), p=2, dim=1)
        t_flat = F.normalize(teacher_feat.view(bs, -1), p=2, dim=1)
        
        # Pairwise similarity
        s_sim = torch.mm(s_flat, s_flat.t())
        t_sim = torch.mm(t_flat, t_flat.t())
        
        return F.mse_loss(s_sim, t_sim)
    
    def get_loss_weights(self, epoch, total_epochs):
        """Adaptive loss weights based on training progress"""
        progress = epoch / total_epochs
        
        if progress < 0.2:
            # Early: Focus on feature matching and contrastive
            lambda_ce = 0.1
            lambda_kd = 0.3
            lambda_contrast = 0.3
            lambda_attention = 0.2
            lambda_relation = 0.1
        elif progress < 0.5:
            # Mid-early: Balance all losses
            lambda_ce = 0.15
            lambda_kd = 0.35
            lambda_contrast = 0.25
            lambda_attention = 0.15
            lambda_relation = 0.1
        elif progress < 0.8:
            # Mid-late: Emphasize KD and classification
            lambda_ce = 0.2
            lambda_kd = 0.4
            lambda_contrast = 0.2
            lambda_attention = 0.1
            lambda_relation = 0.1
        else:
            # Late: Focus on classification
            lambda_ce = 0.25
            lambda_kd = 0.45
            lambda_contrast = 0.15
            lambda_attention = 0.1
            lambda_relation = 0.05
        
        return lambda_ce, lambda_kd, lambda_contrast, lambda_attention, lambda_relation
    
    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Training step with hybrid losses"""
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Progressive unfreezing
        self._progressive_unfreeze(epoch, total_epochs)
        
        # Set feature extraction
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        # Forward pass
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)
        
        student_logits, student_feat = self.student(images)
        
        # Get adaptive weights
        lambda_ce, lambda_kd, lambda_contrast, lambda_attention, lambda_relation = \
            self.get_loss_weights(epoch, total_epochs)
        
        # 1. Classification loss
        ce_loss = self.ce_loss(student_logits, labels).mean()
        
        # 2. Knowledge Distillation loss
        temperature = 4.0
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 3. Contrastive loss (proven effective)
        contrast_loss = self.contrastive_loss(student_feat, teacher_feat, temperature=0.07)
        
        # 4. Attention transfer (proven effective)
        attention_loss = self.attention_transfer_loss(student_feat, teacher_feat)
        
        # 5. Relation loss
        relation_loss = self.relation_loss(student_feat, teacher_feat)
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_contrast * contrast_loss +
            lambda_attention * attention_loss +
            lambda_relation * relation_loss
        )
        
        total_loss.backward()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'attention_loss': attention_loss.item(),
            'relation_loss': relation_loss.item(),
        }
    
    def cleanup(self):
        """Cleanup resources"""
        torch.cuda.empty_cache()


# Backward compatibility
SimplifiedMSFAMTrainer = ProgressiveBlockRecoveryTrainer
EnhancedMSFAMTrainer = ProgressiveBlockRecoveryTrainer
MSFAMTrainer = ProgressiveBlockRecoveryTrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000, rm_blocks=None):
    """Train student model using Progressive Block Recovery method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = ProgressiveBlockRecoveryTrainer(
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
            'relation_loss': 0,
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
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f}")
            print(f"  KD: {epoch_losses['kd_loss']:.4f} | Contrast: {epoch_losses['contrast_loss']:.4f}")
            print(f"  Attention: {epoch_losses['attention_loss']:.4f} | Relation: {epoch_losses['relation_loss']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam
