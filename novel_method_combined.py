
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance


class MMDFeatureLoss(nn.Module):
    """Maximum Mean Discrepancy for feature distribution matching"""
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class RelationAwareKD(nn.Module):
    """Relation-aware knowledge distillation using feature relationships"""
    def __init__(self):
        super().__init__()
    
    def forward(self, student_feat, teacher_feat):
        bs = student_feat.size(0)
        student_feat_flat = student_feat.view(bs, -1)
        teacher_feat_flat = teacher_feat.view(bs, -1)
        
        student_norm = F.normalize(student_feat_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat_flat, p=2, dim=1)
        
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        loss = F.mse_loss(student_sim, teacher_sim)
        return loss


class AttentionTransferLoss(nn.Module):
    """Attention transfer loss between teacher and student"""
    def __init__(self):
        super().__init__()
    
    def attention_map(self, feature):
        return torch.sum(feature.pow(2), dim=1, keepdim=True)
    
    def forward(self, student_feat, teacher_feat):
        s_attention = self.attention_map(student_feat)
        t_attention = self.attention_map(teacher_feat)
        
        s_attention = F.normalize(s_attention.view(s_attention.size(0), -1), p=2, dim=1)
        t_attention = F.normalize(t_attention.view(t_attention.size(0), -1), p=2, dim=1)
        
        loss = F.mse_loss(s_attention, t_attention)
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature alignment"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_feat, teacher_feat):
        batch_size = student_feat.size(0)
        
        student_feat = F.normalize(student_feat.view(batch_size, -1), dim=1)
        teacher_feat = F.normalize(teacher_feat.view(batch_size, -1), dim=1)
        
        logits = torch.mm(student_feat, teacher_feat.t()) / self.temperature
        labels = torch.arange(batch_size).to(student_feat.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class CombinedKDTrainer:
    """Combined trainer using MMD, Relation-aware, Attention Transfer, and Contrastive KD"""
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        # Initialize all loss functions
        self.mmd_loss = MMDFeatureLoss()
        self.relation_loss = RelationAwareKD()
        self.attention_loss = AttentionTransferLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.teacher.eval()
    
    def train_step(self, images, labels, optimizer):
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        # Forward pass
        student_logits, student_features = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(images)
        
        # Flatten features for distribution-based losses
        s_feat_flat = student_features.view(student_features.size(0), -1)
        t_feat_flat = teacher_features.view(teacher_features.size(0), -1)
        
        # Compute all losses
        # 1. MMD loss - distribution matching
        mmd_loss = self.mmd_loss(s_feat_flat, t_feat_flat)
        
        # 2. Relation loss - pairwise relationships
        relation_loss = self.relation_loss(student_features, teacher_features)
        
        # 3. Attention transfer - spatial attention
        attention_loss = self.attention_loss(student_features, teacher_features)
        
        # 4. Contrastive loss - instance-level alignment
        contrastive_loss = self.contrastive_loss(student_features, teacher_features)
        
        # 5. Knowledge distillation loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / 3.0, dim=1),
            F.softmax(teacher_logits / 3.0, dim=1)
        ) * 9.0
        
        # 6. Classification loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss with balanced weights
        # Distribute weights: KD (25%), CE (15%), MMD (15%), Relation (15%), Attention (15%), Contrastive (15%)
        total_loss = (
            0.25 * kd_loss +
            0.15 * ce_loss +
            0.15 * mmd_loss +
            0.15 * relation_loss +
            0.15 * attention_loss +
            0.15 * contrastive_loss
        )
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'mmd_loss': mmd_loss.item(),
            'relation_loss': relation_loss.item(),
            'attention_loss': attention_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }


def train_with_combined_method(student_model, teacher_model, train_loader, epochs=100, lr=0.01):
    """Train student model using combined KD method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = CombinedKDTrainer(student_model, teacher_model, device)
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'kd_loss': 0,
            'ce_loss': 0,
            'mmd_loss': 0,
            'relation_loss': 0,
            'attention_loss': 0,
            'contrastive_loss': 0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            losses = trainer.train_step(images, labels, optimizer)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        scheduler.step()
        
        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(" - ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
    
    return student_model
