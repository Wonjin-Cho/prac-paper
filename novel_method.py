
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features from multiple layers for multi-scale alignment"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        
        # Extract from multiple scales
        target_layers = [
            'layer1.0.conv1', 'layer1.2.conv2',
            'layer2.0.conv1', 'layer2.3.conv2',
            'layer3.0.conv1', 'layer3.5.conv2',
            'layer4.0.conv1', 'layer4.2.conv2'
        ]
        
        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self.save_feature(name))
                self.hooks.append(hook)
    
    def save_feature(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class WassersteinFeatureAlignment(nn.Module):
    """Wasserstein distance-based feature alignment"""
    def __init__(self):
        super().__init__()
    
    def forward(self, student_feat, teacher_feat):
        # Flatten features
        s_flat = student_feat.view(student_feat.size(0), -1)
        t_flat = teacher_feat.view(teacher_feat.size(0), -1)
        
        # Normalize to unit variance
        s_norm = (s_flat - s_flat.mean(dim=0)) / (s_flat.std(dim=0) + 1e-8)
        t_norm = (t_flat - t_flat.mean(dim=0)) / (t_flat.std(dim=0) + 1e-8)
        
        # Compute sliced Wasserstein distance (approximation)
        num_projections = 50
        losses = []
        
        for _ in range(num_projections):
            # Random projection direction
            direction = torch.randn(s_norm.size(1), device=s_norm.device)
            direction = direction / direction.norm()
            
            # Project features
            s_proj = torch.matmul(s_norm, direction)
            t_proj = torch.matmul(t_norm, direction)
            
            # Sort projections
            s_sorted, _ = torch.sort(s_proj)
            t_sorted, _ = torch.sort(t_proj)
            
            # Wasserstein-1 distance
            losses.append(torch.mean(torch.abs(s_sorted - t_sorted)))
        
        return torch.mean(torch.stack(losses))


class AdaptiveUncertaintyMixup:
    """Mixup with curriculum-based uncertainty weighting"""
    def __init__(self, teacher_model, alpha=1.0, num_classes=1000):
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.num_classes = num_classes
        self.uncertainty_ema = None
        self.ema_decay = 0.9
    
    def __call__(self, images, labels, epoch_progress=0.0):
        if len(images) < 2:
            return images, labels
        
        batch_size = images.size(0)
        
        # Get teacher predictions to estimate uncertainty
        with torch.no_grad():
            teacher_logits, _ = self.teacher_model(images)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            
            # Uncertainty = entropy of predictions
            uncertainty = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=1)
        
        # Update EMA of uncertainty
        if self.uncertainty_ema is None:
            self.uncertainty_ema = uncertainty.mean()
        else:
            self.uncertainty_ema = self.ema_decay * self.uncertainty_ema + \
                                   (1 - self.ema_decay) * uncertainty.mean()
        
        # Adaptive mixing based on uncertainty and curriculum
        # Early training: less aggressive mixing
        # Later training: more aggressive mixing for hard samples
        base_lambda = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size]).to(images.device)
        
        # Adjust lambda based on uncertainty (relative to running average)
        uncertainty_weight = torch.clamp(uncertainty / (self.uncertainty_ema + 1e-8), 0.5, 1.5)
        adjusted_lambda = base_lambda * uncertainty_weight
        adjusted_lambda = torch.clamp(adjusted_lambda, 0.1, 0.9)
        
        # Curriculum: gradually increase mixing strength
        curriculum_factor = 0.5 + 0.5 * epoch_progress
        adjusted_lambda = adjusted_lambda * curriculum_factor + (1 - curriculum_factor) * 0.5
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = torch.zeros_like(images)
        for i in range(batch_size):
            lam = adjusted_lambda[i].item()
            mixed_images[i] = lam * images[i] + (1 - lam) * images[index[i]]
        
        # Convert hard labels to soft labels if needed
        if len(labels.shape) == 1:
            mixed_labels = torch.zeros(batch_size, self.num_classes, device=labels.device)
            
            for i in range(batch_size):
                lam = adjusted_lambda[i].item()
                soft_label = torch.zeros(self.num_classes, device=labels.device)
                soft_label[labels[i]] = lam
                soft_label[labels[index[i]]] += (1 - lam)
                mixed_labels[i] = soft_label
        else:
            # Already soft labels
            mixed_labels = torch.zeros_like(labels)
            for i in range(batch_size):
                lam = adjusted_lambda[i].item()
                mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[index[i]]
        
        return mixed_images, mixed_labels


class MSFAMTrainer:
    """Enhanced Multi-Scale Feature Alignment with Adaptive Mixup Trainer"""
    def __init__(self, student_model, teacher_model, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Multi-scale feature extractors
        self.student_extractor = MultiScaleFeatureExtractor(self.student)
        self.teacher_extractor = MultiScaleFeatureExtractor(self.teacher)
        
        # Loss functions
        self.wasserstein_align = WassersteinFeatureAlignment()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Adaptive mixup
        self.mixup = AdaptiveUncertaintyMixup(self.teacher, num_classes=num_classes)
        
        self.teacher.eval()
        
        # Track training statistics
        self.running_losses = {
            'kd': 0.0,
            'feature': 0.0,
            'ce': 0.0
        }
    
    def compute_multi_scale_feature_loss(self, images):
        """Compute Wasserstein-based feature alignment at multiple scales"""
        student_features = self.student_extractor(images)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)
        
        total_loss = 0.0
        count = 0
        
        for layer_name in student_features.keys():
            if layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                # Use Wasserstein distance
                loss = self.wasserstein_align(s_feat, t_feat)
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Training step with adaptive mixup and multi-scale alignment"""
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Epoch progress for curriculum learning
        epoch_progress = min(epoch / total_epochs, 1.0)
        
        # Apply adaptive mixup
        mixed_images, mixed_labels = self.mixup(images, labels, epoch_progress)
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        # Forward pass
        student_logits, student_feat = self.student(mixed_images)
        
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)  # Teacher on original images
        
        # 1. KD loss with temperature scaling
        temperature = 4.0 - 1.0 * epoch_progress  # Decrease from 4.0 to 3.0
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 2. Classification loss on mixed labels
        if len(mixed_labels.shape) == 1:
            ce_loss = self.ce_loss(student_logits, mixed_labels)
        else:
            # Soft label cross entropy
            ce_loss = -torch.mean(torch.sum(mixed_labels * F.log_softmax(student_logits, dim=1), dim=1))
        
        # 3. Multi-scale feature alignment (on original images)
        feat_loss = self.compute_multi_scale_feature_loss(images)
        
        # 4. Final layer feature alignment with Wasserstein
        final_feat_loss = self.wasserstein_align(student_feat, teacher_feat)
        
        # Adaptive loss weights with curriculum
        # Early: focus on CE and basic KD
        # Later: increase feature alignment importance
        lambda_ce = 0.3 - 0.1 * epoch_progress  # 0.3 -> 0.2
        lambda_kd = 0.4 + 0.1 * epoch_progress  # 0.4 -> 0.5
        lambda_feat = 0.15 + 0.15 * epoch_progress  # 0.15 -> 0.3
        lambda_final = 0.15
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * feat_loss +
            lambda_final * final_feat_loss
        )
        
        # Backward and optimize
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update running statistics
        self.running_losses['kd'] = 0.9 * self.running_losses['kd'] + 0.1 * kd_loss.item()
        self.running_losses['feature'] = 0.9 * self.running_losses['feature'] + 0.1 * feat_loss.item()
        self.running_losses['ce'] = 0.9 * self.running_losses['ce'] + 0.1 * ce_loss.item()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'feat_loss': feat_loss.item(),
            'final_feat_loss': final_feat_loss.item()
        }
    
    def cleanup(self):
        """Remove hooks"""
        self.student_extractor.remove_hooks()
        self.teacher_extractor.remove_hooks()


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000):
    """Train student model using MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = MSFAMTrainer(student_model, teacher_model, num_classes, device)
    
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
            'ce_loss': 0,
            'kd_loss': 0,
            'feat_loss': 0,
            'final_feat_loss': 0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale
            
            losses = trainer.train_step(images, labels, optimizer, epoch, epochs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
            print(f"  CE Loss: {epoch_losses['ce_loss']:.4f}")
            print(f"  KD Loss: {epoch_losses['kd_loss']:.4f}")
            print(f"  Feature Loss: {epoch_losses['feat_loss']:.4f}")
            print(f"  Final Feature Loss: {epoch_losses['final_feat_loss']:.4f}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    trainer.cleanup()
    return student_model


# Backward compatibility
SimplifiedMSFAMTrainer = MSFAMTrainer
train_with_simplified_msfam = train_with_msfam
