
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # Compute pairwise relationships
        bs = student_feat.size(0)
        student_feat_flat = student_feat.view(bs, -1)
        teacher_feat_flat = teacher_feat.view(bs, -1)
        
        # Normalize features
        student_norm = F.normalize(student_feat_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat_flat, p=2, dim=1)
        
        # Compute similarity matrices
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        # Loss is the difference between similarity matrices
        loss = F.mse_loss(student_sim, teacher_sim)
        return loss


class MMDKDTrainer:
    """Trainer using MMD and Relation-aware KD"""
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        self.mmd_loss = MMDFeatureLoss()
        self.relation_loss = RelationAwareKD()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.teacher.eval()
    
    def train_step(self, images, labels, optimizer):
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Get features
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        student_logits, student_features = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(images)
        
        # Flatten features for MMD
        s_feat_flat = student_features.view(student_features.size(0), -1)
        t_feat_flat = teacher_features.view(teacher_features.size(0), -1)
        
        # MMD loss
        mmd_loss = self.mmd_loss(s_feat_flat, t_feat_flat)
        
        # Relation loss
        relation_loss = self.relation_loss(student_features, teacher_features)
        
        # KD loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / 3.0, dim=1),
            F.softmax(teacher_logits / 3.0, dim=1)
        ) * 9.0
        
        # CE loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = 0.3 * kd_loss + 0.2 * ce_loss + 0.3 * mmd_loss + 0.2 * relation_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'mmd_loss': mmd_loss.item(),
            'relation_loss': relation_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }
