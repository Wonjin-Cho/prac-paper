
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardNegativeContrastiveLoss(nn.Module):
    """
    Enhanced contrastive loss with hard negative mining.
    
    Key improvements over basic contrastive:
    1. Hard negative mining: Focus on most confusing negative pairs
    2. Momentum queue: Maintain larger set of negative samples
    3. Temperature scheduling: Adaptive temperature based on difficulty
    """
    def __init__(self, temperature=0.07, queue_size=4096, hard_negative_ratio=0.5):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.hard_negative_ratio = hard_negative_ratio
        
        # Momentum queue for negative samples (teacher features)
        self.register_buffer("queue", torch.randn(queue_size, 512))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new teacher features"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest samples in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remain = self.queue_size - ptr
            self.queue[ptr:] = keys[:remain]
            self.queue[:batch_size - remain] = keys[remain:]
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, student_feat, teacher_feat):
        batch_size = student_feat.size(0)
        
        # Flatten and normalize features
        student_feat = F.normalize(student_feat.view(batch_size, -1), dim=1)
        teacher_feat = F.normalize(teacher_feat.view(batch_size, -1), dim=1)
        
        # Positive pairs: student[i] with teacher[i]
        pos_sim = torch.sum(student_feat * teacher_feat, dim=1, keepdim=True)
        
        # Negative pairs: student[i] with all other teachers in batch
        neg_sim_batch = torch.mm(student_feat, teacher_feat.t())
        # Remove diagonal (positive pairs)
        mask = torch.eye(batch_size, dtype=torch.bool, device=student_feat.device)
        neg_sim_batch = neg_sim_batch.masked_fill(mask, -1e9)
        
        # Negative pairs: student[i] with queue
        neg_sim_queue = torch.mm(student_feat, self.queue.t())
        
        # Combine all negative similarities
        all_neg_sim = torch.cat([neg_sim_batch, neg_sim_queue], dim=1)
        
        # Hard negative mining: select top-k hardest negatives
        num_negatives = all_neg_sim.size(1)
        num_hard = int(num_negatives * self.hard_negative_ratio)
        
        # Get hardest negatives (highest similarity = hardest)
        hard_neg_sim, _ = torch.topk(all_neg_sim, k=num_hard, dim=1)
        
        # Combine positive and hard negative similarities
        logits = torch.cat([pos_sim, hard_neg_sim], dim=1) / self.temperature
        
        # Labels: first position is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=student_feat.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Update queue with current teacher features
        with torch.no_grad():
            self._dequeue_and_enqueue(teacher_feat)
        
        return loss


class MomentumEncoder(nn.Module):
    """
    Momentum-updated encoder for stable target generation.
    Prevents teacher from changing too rapidly during training.
    """
    def __init__(self, base_encoder, momentum=0.999):
        super().__init__()
        self.encoder = base_encoder
        self.momentum = momentum
        
        # Initialize momentum encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, student_encoder):
        """Update momentum encoder with student parameters"""
        for param_m, param_s in zip(self.encoder.parameters(), student_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_s.data * (1.0 - self.momentum)
    
    def forward(self, x):
        return self.encoder(x)


class EnhancedContrastiveTrainer:
    """
    Enhanced contrastive trainer with hard negative mining and momentum encoder.
    
    This keeps the core contrastive learning approach that worked (66.04%)
    but adds sophisticated improvements from recent contrastive learning research.
    """
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        # Momentum encoder for stable targets
        self.momentum_teacher = MomentumEncoder(teacher_model, momentum=0.999)
        
        # Enhanced contrastive loss with hard negative mining
        self.contrastive_loss = HardNegativeContrastiveLoss(
            temperature=0.07,
            queue_size=4096,
            hard_negative_ratio=0.5
        )
        
        # Standard losses
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.teacher.eval()
        self.momentum_teacher.encoder.eval()
    
    def train_step(self, images, labels, optimizer, epoch=0, total_epochs=100):
        """Single training step with adaptive loss weighting"""
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.momentum_teacher.encoder.get_feat = 'pre_GAP'
        
        # Student forward pass
        student_logits, student_features = self.student(images)
        
        # Momentum teacher forward pass (for stable contrastive targets)
        with torch.no_grad():
            teacher_logits, teacher_features = self.momentum_teacher(images)
        
        # Hard negative contrastive loss
        contrast_loss = self.contrastive_loss(student_features, teacher_features)
        
        # Knowledge distillation loss
        temperature = 4.0
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # Classification loss with label smoothing
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Adaptive loss weighting based on training progress
        progress = epoch / total_epochs
        if progress < 0.3:
            # Early: focus on feature alignment
            lambda_contrast = 0.6
            lambda_kd = 0.3
            lambda_ce = 0.1
        elif progress < 0.7:
            # Mid: balanced
            lambda_contrast = 0.4
            lambda_kd = 0.4
            lambda_ce = 0.2
        else:
            # Late: focus on classification
            lambda_contrast = 0.3
            lambda_kd = 0.4
            lambda_ce = 0.3
        
        total_loss = lambda_contrast * contrast_loss + lambda_kd * kd_loss + lambda_ce * ce_loss
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update momentum encoder
        self.momentum_teacher.update(self.student)
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }


def train_with_enhanced_contrastive(student_model, teacher_model, train_loader, epochs=100, lr=0.01):
    """
    Train student model using enhanced contrastive method with hard negative mining.
    
    This maintains the contrastive approach that worked (66.04%) while adding:
    1. Hard negative mining for better sample selection
    2. Momentum encoder for stable targets
    3. Larger negative sample pool via queue
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = EnhancedContrastiveTrainer(student_model, teacher_model, device)
    
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
            'contrastive_loss': 0,
            'kd_loss': 0,
            'ce_loss': 0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Learning rate warmup
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale
            
            losses = trainer.train_step(images, labels, optimizer, epoch, epochs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} (LR: {current_lr:.6f})")
            print(" - ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
    
    return student_model
