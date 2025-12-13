
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature alignment"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_feat, teacher_feat):
        batch_size = student_feat.size(0)
        
        # Flatten and normalize features
        student_feat = F.normalize(student_feat.view(batch_size, -1), dim=1)
        teacher_feat = F.normalize(teacher_feat.view(batch_size, -1), dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(student_feat, teacher_feat.t()) / self.temperature
        
        # Labels are diagonal (each student should match its corresponding teacher)
        labels = torch.arange(batch_size).to(student_feat.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class MomentumTeacher:
    """Momentum-updated teacher for consistent targets"""
    def __init__(self, teacher_model, momentum=0.999):
        self.teacher = teacher_model
        self.momentum = momentum
    
    def update(self, student_model):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), student_model.parameters()):
                teacher_param.data = self.momentum * teacher_param.data + (1 - self.momentum) * student_param.data


class ContrastiveKDTrainer:
    """Trainer with contrastive learning"""
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.teacher.eval()
    
    def train_step(self, images, labels, optimizer):
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        student_logits, student_features = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(images)
        
        # Contrastive loss
        contrast_loss = self.contrastive_loss(student_features, teacher_features)
        
        # KD loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / 3.0, dim=1),
            F.softmax(teacher_logits / 3.0, dim=1)
        ) * 9.0
        
        # CE loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        total_loss = 0.4 * kd_loss + 0.2 * ce_loss + 0.4 * contrast_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }
