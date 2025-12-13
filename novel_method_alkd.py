
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelRecalibration(nn.Module):
    """Lightweight channel recalibration module using squeeze-excitation"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveFeatureAlignment(nn.Module):
    """Adaptive feature alignment with recalibration"""
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        # Channel alignment if dimensions differ
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
        else:
            self.align = nn.Identity()
        
        # Recalibration module
        self.recalibrate = ChannelRecalibration(teacher_channels)
        
    def forward(self, student_feat, teacher_feat):
        # Align channels
        aligned = self.align(student_feat)
        # Recalibrate
        recalibrated = self.recalibrate(aligned)
        return recalibrated


class LayerWiseDistillationLoss(nn.Module):
    """Layer-wise knowledge distillation with adaptive weights"""
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, layer_weight=1.0):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        return loss * layer_weight


class CurriculumSampler:
    """Sample batches based on difficulty - easier samples first"""
    def __init__(self):
        self.difficulties = {}
        
    def compute_difficulty(self, teacher_logits, labels):
        """Compute sample difficulty based on teacher's confidence"""
        probs = F.softmax(teacher_logits, dim=1)
        confidence = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
        # Lower confidence = harder sample
        difficulty = 1.0 - confidence
        return difficulty
    
    def get_curriculum_weight(self, difficulty, current_epoch, total_epochs):
        """Get weight for each sample based on curriculum"""
        progress = current_epoch / total_epochs
        
        # Early epochs: focus on easy samples (low difficulty)
        # Later epochs: include harder samples
        threshold = 0.3 + 0.7 * progress
        
        # Soft weighting instead of hard filtering
        weights = torch.sigmoid(5 * (threshold - difficulty))
        return weights


class ALKDCRTrainer:
    """Adaptive Layer-wise Knowledge Distillation with Channel Recalibration"""
    def __init__(self, student_model, teacher_model, rm_blocks, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.rm_blocks = rm_blocks if isinstance(rm_blocks, list) else [rm_blocks]
        self.device = device
        
        # Get feature dimensions
        self.student_channels = self._get_feature_channels()
        self.teacher_channels = self._get_feature_channels(is_teacher=True)
        
        # Feature alignment modules
        self.feature_aligners = nn.ModuleDict()
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            s_ch = self.student_channels.get(layer_name, 512)
            t_ch = self.teacher_channels.get(layer_name, 512)
            self.feature_aligners[layer_name] = AdaptiveFeatureAlignment(s_ch, t_ch).to(device)
        
        # Loss functions
        self.kd_loss = LayerWiseDistillationLoss(temperature=4.0)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss for curriculum
        self.mse_loss = nn.MSELoss()
        
        # Curriculum sampler
        self.curriculum = CurriculumSampler()
        
        # Layer weights based on proximity to pruned blocks
        self.layer_weights = self._compute_layer_weights()
        
        self.teacher.eval()
        
    def _get_feature_channels(self, is_teacher=False):
        """Get feature channels for each layer"""
        model = self.teacher if is_teacher else self.student
        channels = {}
        
        # For ResNet architectures
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
            
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(base_model, layer_name):
                layer = getattr(base_model, layer_name)
                # Get output channels from last block
                if len(layer) > 0:
                    last_block = layer[-1]
                    if hasattr(last_block, 'conv2'):
                        channels[layer_name] = last_block.conv2.out_channels
                    elif hasattr(last_block, 'conv3'):
                        channels[layer_name] = last_block.conv3.out_channels
        
        return channels
    
    def _compute_layer_weights(self):
        """Compute adaptive weights for each layer based on pruned blocks"""
        weights = {'layer1': 1.0, 'layer2': 1.0, 'layer3': 1.0, 'layer4': 1.0}
        
        # Increase weight for layers containing pruned blocks
        for rm_block in self.rm_blocks:
            if 'layer1' in rm_block:
                weights['layer1'] = 1.5
                weights['layer2'] = 1.3  # Adjacent layer
            elif 'layer2' in rm_block:
                weights['layer1'] = 1.2
                weights['layer2'] = 1.5
                weights['layer3'] = 1.3
            elif 'layer3' in rm_block:
                weights['layer2'] = 1.2
                weights['layer3'] = 1.5
                weights['layer4'] = 1.3
            elif 'layer4' in rm_block:
                weights['layer3'] = 1.3
                weights['layer4'] = 1.5
        
        return weights
    
    def extract_layer_features(self, model, x):
        """Extract features from each layer"""
        features = {}
        
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        # Forward through initial layers
        x = base_model.conv1(x)
        x = base_model.bn1(x)
        x = base_model.relu(x)
        x = base_model.maxpool(x)
        
        # Extract layer-wise features
        x = base_model.layer1(x)
        features['layer1'] = x
        
        x = base_model.layer2(x)
        features['layer2'] = x
        
        x = base_model.layer3(x)
        features['layer3'] = x
        
        x = base_model.layer4(x)
        features['layer4'] = x
        
        # Get final output
        x = base_model.avgpool(x)
        x = torch.flatten(x, 1)
        logits = base_model.fc(x)
        
        return logits, features
    
    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        self.student.train()
        for aligner in self.feature_aligners.values():
            aligner.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Extract multi-layer features
        student_logits, student_features = self.extract_layer_features(self.student, images)
        
        with torch.no_grad():
            teacher_logits, teacher_features = self.extract_layer_features(self.teacher, images)
            # Compute sample difficulties for curriculum learning
            difficulty = self.curriculum.compute_difficulty(teacher_logits, labels)
            curriculum_weights = self.curriculum.get_curriculum_weight(
                difficulty, epoch, total_epochs
            )
        
        # Progressive feature alignment weight
        progress = epoch / total_epochs
        feature_weight = 0.3 + 0.4 * progress  # Start at 0.3, end at 0.7
        
        # Compute layer-wise feature alignment losses
        total_feature_loss = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if layer_name in student_features and layer_name in teacher_features:
                # Align and recalibrate student features
                aligned_student = self.feature_aligners[layer_name](
                    student_features[layer_name],
                    teacher_features[layer_name]
                )
                
                # MSE loss between aligned student and teacher features
                feat_loss = self.mse_loss(aligned_student, teacher_features[layer_name])
                
                # Weight by layer importance
                layer_weight = self.layer_weights.get(layer_name, 1.0)
                total_feature_loss += feat_loss * layer_weight
        
        # Average feature loss across layers
        total_feature_loss = total_feature_loss / len(self.layer_weights)
        
        # Knowledge distillation loss
        kd_loss = self.kd_loss(student_logits, teacher_logits)
        
        # Classification loss with curriculum weighting
        ce_loss_per_sample = self.ce_loss(student_logits, labels)
        ce_loss = (ce_loss_per_sample * curriculum_weights).mean()
        
        # Adaptive loss composition
        # Early training: focus on classification and basic KD
        # Later training: increase feature alignment importance
        if progress < 0.3:
            # Early phase: learn basics
            lambda_ce = 0.4
            lambda_kd = 0.4
            lambda_feat = 0.2
        elif progress < 0.7:
            # Middle phase: balance
            lambda_ce = 0.3
            lambda_kd = 0.4
            lambda_feat = 0.3
        else:
            # Late phase: fine-tune features
            lambda_ce = 0.2
            lambda_kd = 0.4
            lambda_feat = 0.4
        
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * total_feature_loss
        )
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(
            [p for aligner in self.feature_aligners.values() for p in aligner.parameters()],
            max_norm=5.0
        )
        
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'feature_loss': total_feature_loss.item(),
            'avg_difficulty': difficulty.mean().item()
        }
    
    def get_trainable_parameters(self):
        """Get all trainable parameters including aligners"""
        params = []
        # Student parameters
        params.extend(list(self.student.parameters()))
        # Feature aligner parameters
        for aligner in self.feature_aligners.values():
            params.extend(list(aligner.parameters()))
        return params


def train_with_alkd(student_model, teacher_model, train_loader, rm_blocks, epochs=2000, lr=0.01):
    """Train using ALKD-CR method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = ALKDCRTrainer(student_model, teacher_model, rm_blocks, device)
    
    # Optimizer for both student and feature aligners
    optimizer = torch.optim.SGD(
        trainer.get_trainable_parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    
    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'kd_loss': 0,
            'ce_loss': 0,
            'feature_loss': 0,
            'avg_difficulty': 0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            losses = trainer.train_step(images, labels, optimizer, epoch, epochs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        scheduler.step()
        
        # Print statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(" - ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
    
    return student_model
