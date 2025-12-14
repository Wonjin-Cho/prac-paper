
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales from the model"""
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        for name, module in model.named_modules():
            if name in layer_names:
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


class WassersteinFeatureLoss(nn.Module):
    """Wasserstein distance-based feature alignment"""
    def __init__(self, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, student_feat, teacher_feat):
        # Flatten spatial dimensions
        bs, c, h, w = student_feat.shape
        student_flat = student_feat.view(bs, c, -1)
        teacher_flat = teacher_feat.view(bs, c, -1)
        
        # Compute channel-wise Wasserstein distance
        loss = 0
        for i in range(min(c, self.num_samples)):
            s_dist = student_flat[:, i, :].flatten().detach().cpu().numpy()
            t_dist = teacher_flat[:, i, :].flatten().detach().cpu().numpy()
            
            # Sample for efficiency
            if len(s_dist) > 1000:
                indices = np.random.choice(len(s_dist), 1000, replace=False)
                s_dist = s_dist[indices]
                t_dist = t_dist[indices]
            
            loss += wasserstein_distance(s_dist, t_dist)
        
        return torch.tensor(loss / min(c, self.num_samples), device=student_feat.device)


class AdaptiveUncertaintyMixup:
    """Mixup based on feature uncertainty from teacher model"""
    def __init__(self, teacher_model, alpha=1.0):
        self.teacher_model = teacher_model
        self.alpha = alpha
    
    def compute_uncertainty(self, images):
        """Compute prediction uncertainty using entropy"""
        with torch.no_grad():
            logits = self.teacher_model(images)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        
        # Compute uncertainty for each sample
        uncertainty = self.compute_uncertainty(images)
        
        # Generate mixup weights based on uncertainty
        # High uncertainty samples get lower mixing weights
        weights = torch.softmax(-uncertainty, dim=0)
        
        # Create mixup pairs based on uncertainty
        indices = torch.randperm(batch_size)
        mixed_images = images.clone()
        mixed_labels = labels.clone()
        
        for i in range(batch_size):
            # Lambda from beta distribution, modulated by uncertainty
            lam = np.random.beta(self.alpha, self.alpha)
            lam = lam * (1 - weights[i].item())  # Reduce mixing for uncertain samples
            
            mixed_images[i] = lam * images[i] + (1 - lam) * images[indices[i]]
            mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[indices[i]]
        
        return mixed_images, mixed_labels


class MSFAMTrainer:
    """Multi-Scale Feature Alignment with Adaptive Mixup Trainer"""
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        
        # Extract layer names for multi-scale features
        self.layer_names = self._get_layer_names()
        
        # Feature extractors
        self.student_extractor = MultiScaleFeatureExtractor(
            self.student, self.layer_names
        )
        self.teacher_extractor = MultiScaleFeatureExtractor(
            self.teacher, self.layer_names
        )
        
        # Loss functions
        self.wasserstein_loss = WassersteinFeatureLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Adaptive mixup
        self.mixup = AdaptiveUncertaintyMixup(self.teacher)
        
        self.teacher.eval()
    
    def _get_layer_names(self):
        """Get names of key layers for feature extraction"""
        layer_names = []
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d) and 'downsample' not in name:
                if any(f'layer{i}' in name for i in range(1, 5)):
                    # Extract one layer per block
                    if name.endswith('.conv2'):
                        layer_names.append(name)
        return layer_names[:8]  # Limit to 8 layers for efficiency
    
    def compute_feature_alignment_loss(self, images):
        """Compute multi-scale feature alignment loss"""
        student_features = self.student_extractor(images)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)
        
        feature_loss = 0
        for layer_name in self.layer_names:
            if layer_name in student_features and layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                # L2 loss
                l2_loss = F.mse_loss(s_feat, t_feat)
                
                # Wasserstein loss (computed periodically for efficiency)
                if np.random.random() < 0.1:  # 10% of the time
                    w_loss = self.wasserstein_loss(s_feat, t_feat)
                else:
                    w_loss = 0
                
                feature_loss += l2_loss + 0.01 * w_loss
        
        return feature_loss / len(self.layer_names)
    
    def train_step(self, images, labels, optimizer, use_mixup=True):
        """Single training step"""
        self.student.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Apply adaptive mixup
        if use_mixup:
            images, labels = self.mixup(images, labels)
        
        optimizer.zero_grad()
        
        # Student predictions
        student_logits = self.student(images)
        
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        # Knowledge distillation loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / 3.0, dim=1),
            F.softmax(teacher_logits / 3.0, dim=1)
        ) * (3.0 ** 2)
        
        # Classification loss
        if len(labels.shape) == 2:  # Soft labels from mixup
            ce_loss = -torch.mean(torch.sum(labels * F.log_softmax(student_logits, dim=1), dim=1))
        else:
            ce_loss = self.ce_loss(student_logits, labels)
        
        # Feature alignment loss
        feat_loss = self.compute_feature_alignment_loss(images)
        
        # Combined loss with adaptive weighting
        total_loss = 0.5 * kd_loss + 0.3 * ce_loss + 0.2 * feat_loss
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'feat_loss': feat_loss.item()
        }
    
    def cleanup(self):
        """Remove hooks"""
        self.student_extractor.remove_hooks()
        self.teacher_extractor.remove_hooks()


def train_with_msfam(student_model, teacher_model, train_loader, epochs=100, lr=0.01):
    """Train student model using MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = MSFAMTrainer(student_model, teacher_model, device)
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        epoch_losses = {'total_loss': 0, 'kd_loss': 0, 'ce_loss': 0, 'feat_loss': 0}
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            losses = trainer.train_step(images, labels, optimizer, use_mixup=True)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        scheduler.step()
        
        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - " + 
                  " - ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
    
    trainer.cleanup()
    return student_model
