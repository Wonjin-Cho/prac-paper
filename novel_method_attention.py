
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out)
        return out.view(b, c, 1, 1)


class AttentionTransferLoss(nn.Module):
    """Attention transfer loss between teacher and student"""
    def __init__(self):
        super().__init__()
    
    def attention_map(self, feature):
        # Spatial attention: sum over channels
        return torch.sum(feature.pow(2), dim=1, keepdim=True)
    
    def forward(self, student_feat, teacher_feat):
        s_attention = self.attention_map(student_feat)
        t_attention = self.attention_map(teacher_feat)
        
        # Normalize attention maps
        s_attention = F.normalize(s_attention.view(s_attention.size(0), -1), p=2, dim=1)
        t_attention = F.normalize(t_attention.view(t_attention.size(0), -1), p=2, dim=1)
        
        loss = F.mse_loss(s_attention, t_attention)
        return loss


class AttentionKDTrainer:
    """Trainer with channel and spatial attention transfer"""
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        self.attention_loss = AttentionTransferLoss()
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
        
        # Attention transfer
        att_loss = self.attention_loss(student_features, teacher_features)
        
        # Feature loss
        feat_loss = F.mse_loss(student_features, teacher_features)
        
        # KD loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / 3.0, dim=1),
            F.softmax(teacher_logits / 3.0, dim=1)
        ) * 9.0
        
        # CE loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        total_loss = 0.3 * kd_loss + 0.2 * ce_loss + 0.3 * feat_loss + 0.2 * att_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'attention_loss': att_loss.item(),
            'feature_loss': feat_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }
