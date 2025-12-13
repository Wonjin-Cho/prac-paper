
"""
Adaptive Multi-Scale Feature Reconstruction with Contrastive Regularization (AMFR-CR)

Novel contributions:
1. Multi-scale feature pyramid matching across multiple layers
2. Contrastive loss to preserve semantic relationships
3. Dynamic importance weighting based on block sensitivity
4. Hierarchical knowledge distillation with attention
5. Self-supervised auxiliary tasks for synthetic data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleFeatureMatcher(nn.Module):
    """Match features at multiple scales using learnable projections"""
    
    def __init__(self, teacher_channels, student_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.projections = nn.ModuleList()
        
        for i in range(num_scales):
            # Learnable projection to align teacher and student features
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(student_channels[i], teacher_channels[i], 1),
                    nn.BatchNorm2d(teacher_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: List of feature maps from student [layer1, layer2, layer3]
            teacher_features: List of feature maps from teacher [layer1, layer2, layer3]
        """
        total_loss = 0
        for i, (s_feat, t_feat, proj) in enumerate(zip(
            student_features, teacher_features, self.projections
        )):
            # Project student features
            s_aligned = proj(s_feat)
            
            # Multi-scale matching with spatial attention
            attention = self._compute_spatial_attention(t_feat)
            
            # Weighted MSE loss
            loss = F.mse_loss(s_aligned * attention, t_feat * attention)
            total_loss += loss * (2 ** i)  # Weight later layers more
        
        return total_loss / len(student_features)
    
    def _compute_spatial_attention(self, features):
        """Compute spatial attention map"""
        # Channel-wise mean
        attention = features.mean(dim=1, keepdim=True)
        # Normalize
        attention = torch.sigmoid(attention)
        return attention


class ContrastiveLoss(nn.Module):
    """Contrastive loss to preserve inter-class relationships"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_features, teacher_features, labels):
        """
        Args:
            student_features: (batch_size, feature_dim)
            teacher_features: (batch_size, feature_dim)
            labels: (batch_size,)
        """
        batch_size = student_features.size(0)
        
        # Normalize features
        student_features = F.normalize(student_features, dim=1)
        teacher_features = F.normalize(teacher_features, dim=1)
        
        # Compute similarity matrices
        student_sim = torch.mm(student_features, student_features.t()) / self.temperature
        teacher_sim = torch.mm(teacher_features, teacher_features.t()) / self.temperature
        
        # Create positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask
        
        # KL divergence between student and teacher similarity distributions
        loss = F.kl_div(
            F.log_softmax(student_sim, dim=1),
            F.softmax(teacher_sim, dim=1),
            reduction='batchmean'
        )
        
        return loss


class AdaptiveBlockWeighting(nn.Module):
    """Dynamically weight blocks based on their importance"""
    
    def __init__(self, num_blocks):
        super().__init__()
        # Learnable importance weights
        self.weights = nn.Parameter(torch.ones(num_blocks))
        
    def forward(self, block_losses):
        """
        Args:
            block_losses: List of losses for each block
        Returns:
            Weighted total loss
        """
        # Softmax to ensure weights sum to 1
        normalized_weights = F.softmax(self.weights, dim=0)
        
        total_loss = 0
        for weight, loss in zip(normalized_weights, block_losses):
            total_loss += weight * loss
        
        return total_loss, normalized_weights


class HierarchicalKnowledgeTransfer(nn.Module):
    """Transfer knowledge hierarchically from multiple teacher layers"""
    
    def __init__(self, teacher_dims, student_dims):
        super().__init__()
        self.num_layers = len(teacher_dims)
        self.attention_modules = nn.ModuleList()
        
        for t_dim, s_dim in zip(teacher_dims, student_dims):
            self.attention_modules.append(
                nn.Sequential(
                    nn.Linear(s_dim, s_dim // 4),
                    nn.ReLU(),
                    nn.Linear(s_dim // 4, t_dim)
                )
            )
    
    def forward(self, student_outputs, teacher_outputs):
        """
        Args:
            student_outputs: List of student layer outputs
            teacher_outputs: List of teacher layer outputs
        """
        total_loss = 0
        
        for s_out, t_out, attn in zip(
            student_outputs, teacher_outputs, self.attention_modules
        ):
            # Global average pooling
            s_pooled = F.adaptive_avg_pool2d(s_out, 1).flatten(1)
            t_pooled = F.adaptive_avg_pool2d(t_out, 1).flatten(1)
            
            # Apply attention
            s_attended = attn(s_pooled)
            
            # MSE loss
            loss = F.mse_loss(s_attended, t_pooled)
            total_loss += loss
        
        return total_loss / self.num_layers


class SelfSupervisedAuxTask(nn.Module):
    """Auxiliary self-supervised task for synthetic data"""
    
    def __init__(self, feature_dim, num_rotations=4):
        super().__init__()
        self.num_rotations = num_rotations
        self.rotation_predictor = nn.Linear(feature_dim, num_rotations)
    
    def forward(self, features, images):
        """
        Args:
            features: Student features before classification
            images: Input images
        Returns:
            Rotation prediction loss
        """
        batch_size = images.size(0)
        
        # Create rotated versions
        rotated_images = []
        rotation_labels = []
        
        for i in range(self.num_rotations):
            rotated = torch.rot90(images, i, [2, 3])
            rotated_images.append(rotated)
            rotation_labels.extend([i] * batch_size)
        
        rotated_images = torch.cat(rotated_images, dim=0)
        rotation_labels = torch.tensor(rotation_labels).to(images.device)
        
        # Forward pass through student (you'd need to hook this)
        # This is a simplified version - in practice, recompute features
        
        # Predict rotations
        rotation_pred = self.rotation_predictor(features)
        
        # Cross-entropy loss
        loss = F.cross_entropy(rotation_pred, rotation_labels)
        
        return loss


class AMFRCRLoss(nn.Module):
    """Complete AMFR-CR loss combining all components"""
    
    def __init__(self, teacher_channels, student_channels, feature_dim):
        super().__init__()
        
        # Components
        self.ms_matcher = MultiScaleFeatureMatcher(teacher_channels, student_channels)
        self.contrastive = ContrastiveLoss(temperature=0.07)
        self.block_weighting = AdaptiveBlockWeighting(num_blocks=len(student_channels))
        self.hierarchical_kd = HierarchicalKnowledgeTransfer(teacher_channels, student_channels)
        self.aux_task = SelfSupervisedAuxTask(feature_dim)
        
        # Loss weights
        self.alpha_ms = 1.0  # Multi-scale feature matching
        self.alpha_contrast = 0.5  # Contrastive loss
        self.alpha_hier = 0.3  # Hierarchical KD
        self.alpha_aux = 0.1  # Auxiliary task
    
    def forward(self, student_outputs, teacher_outputs, labels, images):
        """
        Args:
            student_outputs: Dict with 'features' (list), 'logits', 'final_features'
            teacher_outputs: Dict with 'features' (list), 'logits', 'final_features'
            labels: Ground truth labels
            images: Input images
        """
        # 1. Multi-scale feature matching
        ms_loss = self.ms_matcher(
            student_outputs['features'],
            teacher_outputs['features']
        )
        
        # 2. Contrastive loss on final features
        contrast_loss = self.contrastive(
            student_outputs['final_features'],
            teacher_outputs['final_features'],
            labels
        )
        
        # 3. Hierarchical knowledge distillation
        hier_loss = self.hierarchical_kd(
            student_outputs['features'],
            teacher_outputs['features']
        )
        
        # 4. Self-supervised auxiliary task
        aux_loss = self.aux_task(
            student_outputs['final_features'],
            images
        )
        
        # 5. Standard KD loss on logits
        T = 4.0
        kd_loss = F.kl_div(
            F.log_softmax(student_outputs['logits'] / T, dim=1),
            F.softmax(teacher_outputs['logits'] / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Combine all losses
        total_loss = (
            self.alpha_ms * ms_loss +
            self.alpha_contrast * contrast_loss +
            self.alpha_hier * hier_loss +
            self.alpha_aux * aux_loss +
            kd_loss
        )
        
        return {
            'total': total_loss,
            'ms_loss': ms_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'hier_loss': hier_loss.item(),
            'aux_loss': aux_loss.item(),
            'kd_loss': kd_loss.item()
        }


def extract_multi_scale_features(model, x, layer_names):
    """
    Extract features from multiple layers of the model
    
    Args:
        model: PyTorch model
        x: Input tensor
        layer_names: List of layer names to extract features from
    
    Returns:
        Dictionary with features and logits
    """
    features = []
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            features.append(output)
        return hook
    
    # Register hooks
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    logits = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get final features (before classifier)
    final_features = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
    
    return {
        'features': features,
        'logits': logits,
        'final_features': final_features
    }
