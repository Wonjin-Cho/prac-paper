
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


class ContrastiveLearningHead(nn.Module):
    """Self-supervised contrastive learning head for better representation"""
    def __init__(self, feature_dim=512, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
    
    def forward(self, features):
        # Global average pooling if needed
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return F.normalize(self.projection(features), dim=1)


class FeatureDiscriminator(nn.Module):
    """Discriminator for adversarial feature alignment"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, features):
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        return self.discriminator(features)


class HardSampleMiner:
    """Mine hard samples for focused training"""
    def __init__(self, percentile=0.7):
        self.percentile = percentile
    
    def get_hard_samples(self, student_logits, teacher_logits, labels):
        """Identify hard samples based on prediction disagreement"""
        with torch.no_grad():
            # KL divergence between student and teacher
            s_probs = F.softmax(student_logits, dim=1)
            t_probs = F.softmax(teacher_logits, dim=1)
            kl_div = torch.sum(t_probs * torch.log(t_probs / (s_probs + 1e-10) + 1e-10), dim=1)
            
            # Get threshold for hard samples
            threshold = torch.quantile(kl_div, self.percentile)
            hard_mask = kl_div > threshold
            
        return hard_mask, kl_div


class MSFAMTrainer:
    """Enhanced Multi-Scale Feature Alignment with Adaptive Mixup Trainer"""
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
        
        # NEW: Contrastive learning head
        self.contrastive_head = ContrastiveLearningHead(feature_dim=512, projection_dim=128).to(device)
        
        # NEW: Feature discriminator for adversarial alignment
        self.discriminator = FeatureDiscriminator(feature_dim=512).to(device)
        
        # NEW: Hard sample miner
        self.hard_miner = HardSampleMiner(percentile=0.7)
        
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
    
    def compute_contrastive_loss(self, features1, features2, temperature=0.5):
        """Compute contrastive loss between two feature sets"""
        # Project features
        z1 = self.contrastive_head(features1)
        z2 = self.contrastive_head(features2)
        
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def compute_adversarial_loss(self, student_features, teacher_features):
        """Compute adversarial loss for feature alignment"""
        # Discriminator tries to distinguish student from teacher
        student_pred = self.discriminator(student_features)
        teacher_pred = self.discriminator(teacher_features.detach())
        
        # Student tries to fool discriminator (make features indistinguishable)
        real_labels = torch.ones_like(student_pred)
        fake_labels = torch.zeros_like(teacher_pred)
        
        # Adversarial loss for student (wants to be classified as "real"/teacher-like)
        student_loss = F.binary_cross_entropy_with_logits(student_pred, real_labels)
        
        # Discriminator loss (distinguish student from teacher)
        disc_loss = (F.binary_cross_entropy_with_logits(student_pred.detach(), fake_labels) +
                     F.binary_cross_entropy_with_logits(teacher_pred, real_labels)) * 0.5
        
        return student_loss, disc_loss
    
    def compute_feature_alignment_loss(self, images):
        """Compute multi-scale feature alignment loss"""
        student_features = self.student_extractor(images)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)
        
        feature_loss = torch.tensor(0.0, device=self.device)
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
                    w_loss = torch.tensor(0.0, device=self.device)
                
                feature_loss += l2_loss + 0.01 * w_loss
        
        return feature_loss / len(self.layer_names)
    
    def train_step(self, images, labels, optimizer, disc_optimizer, epoch, total_epochs, use_mixup=True):
        """Enhanced training step with new components"""
        self.student.train()
        self.contrastive_head.train()
        self.discriminator.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Apply adaptive mixup
        if use_mixup:
            mixed_images, mixed_labels = self.mixup(images, labels)
        else:
            mixed_images, mixed_labels = images, labels
        
        optimizer.zero_grad()
        disc_optimizer.zero_grad()
        
        # Student predictions
        student_logits = self.student(mixed_images)
        
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(mixed_images)
        
        # Get features for additional losses
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        _, student_feat = self.student(mixed_images)
        with torch.no_grad():
            _, teacher_feat = self.teacher(mixed_images)
        
        # NEW: Hard sample mining
        hard_mask, difficulty = self.hard_miner.get_hard_samples(student_logits, teacher_logits, labels)
        hard_weight = torch.ones_like(difficulty)
        hard_weight[hard_mask] = 2.0  # Double weight for hard samples
        hard_weight = hard_weight / hard_weight.sum() * len(difficulty)
        
        # Dynamic temperature based on training progress
        progress = epoch / total_epochs
        temperature = 3.0 + 2.0 * (1 - progress)  # Start at 5, end at 3
        
        # Knowledge distillation loss with dynamic temperature
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # Classification loss with hard sample weighting
        if len(mixed_labels.shape) == 2:  # Soft labels from mixup
            ce_loss = -torch.mean(
                hard_weight * torch.sum(mixed_labels * F.log_softmax(student_logits, dim=1), dim=1)
            )
        else:
            ce_loss = F.cross_entropy(student_logits, mixed_labels, reduction='none')
            ce_loss = (ce_loss * hard_weight).mean()
        
        # Feature alignment loss
        feat_loss = self.compute_feature_alignment_loss(mixed_images)
        
        # NEW: Contrastive loss (self-supervised)
        # Create two augmented views
        contrastive_loss = self.compute_contrastive_loss(student_feat, teacher_feat)
        
        # NEW: Adversarial feature alignment
        adv_student_loss, adv_disc_loss = self.compute_adversarial_loss(student_feat, teacher_feat)
        
        # NEW: Feature consistency regularization
        # Ensure features are consistent across different forward passes
        with torch.no_grad():
            _, student_feat2 = self.student(mixed_images)
        consistency_loss = F.mse_loss(student_feat, student_feat2)
        
        # Adaptive loss composition based on training progress
        if progress < 0.3:
            # Early phase: focus on basic learning
            lambda_ce = 0.4
            lambda_kd = 0.3
            lambda_feat = 0.2
            lambda_contrast = 0.05
            lambda_adv = 0.05
            lambda_consist = 0.0
        elif progress < 0.7:
            # Middle phase: balanced learning
            lambda_ce = 0.25
            lambda_kd = 0.35
            lambda_feat = 0.2
            lambda_contrast = 0.1
            lambda_adv = 0.05
            lambda_consist = 0.05
        else:
            # Late phase: fine-tuning with all components
            lambda_ce = 0.2
            lambda_kd = 0.3
            lambda_feat = 0.2
            lambda_contrast = 0.15
            lambda_adv = 0.1
            lambda_consist = 0.05
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * feat_loss +
            lambda_contrast * contrastive_loss +
            lambda_adv * adv_student_loss +
            lambda_consist * consistency_loss
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Update discriminator
        adv_disc_loss.backward()
        disc_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'feat_loss': feat_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'adv_loss': adv_student_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'hard_samples_pct': hard_mask.float().mean().item()
        }
    
    def cleanup(self):
        """Remove hooks"""
        self.student_extractor.remove_hooks()
        self.teacher_extractor.remove_hooks()


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01):
    """Train student model using enhanced MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = MSFAMTrainer(student_model, teacher_model, device)
    
    # Optimizer for student, contrastive head
    optimizer = torch.optim.SGD(
        list(student_model.parameters()) + list(trainer.contrastive_head.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Separate optimizer for discriminator
    disc_optimizer = torch.optim.Adam(
        trainer.discriminator.parameters(),
        lr=0.0001,
        betas=(0.5, 0.999)
    )
    
    # Cosine annealing scheduler with warmup
    warmup_epochs = int(0.1 * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    
    for epoch in range(epochs):
        epoch_losses = {
            'total_loss': 0,
            'kd_loss': 0,
            'ce_loss': 0,
            'feat_loss': 0,
            'contrastive_loss': 0,
            'adv_loss': 0,
            'consistency_loss': 0,
            'hard_samples_pct': 0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr_scale = (epoch * len(train_loader) + batch_idx + 1) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale
            
            losses = trainer.train_step(
                images, labels, optimizer, disc_optimizer,
                epoch, epochs, use_mixup=True
            )
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Print epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(" - ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    trainer.cleanup()
    return student_model
