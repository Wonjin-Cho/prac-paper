
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales including residuals"""
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


class SlicedWassersteinLoss(nn.Module):
    """Efficient differentiable approximation of Wasserstein distance"""
    def __init__(self, num_projections=50):
        super().__init__()
        self.num_projections = num_projections
    
    def forward(self, student_feat, teacher_feat):
        # Flatten spatial dimensions
        bs, c, h, w = student_feat.shape
        student_flat = student_feat.view(bs, c, -1).transpose(1, 2)  # [bs, hw, c]
        teacher_flat = teacher_flat.view(bs, c, -1).transpose(1, 2)
        
        # Random projections
        device = student_feat.device
        projections = torch.randn(c, self.num_projections, device=device)
        projections = F.normalize(projections, dim=0)
        
        # Project features
        student_proj = torch.matmul(student_flat, projections)  # [bs, hw, num_proj]
        teacher_proj = torch.matmul(teacher_flat, projections)
        
        # Sort and compute L2 distance (differentiable Wasserstein approximation)
        student_sorted, _ = torch.sort(student_proj, dim=1)
        teacher_sorted, _ = torch.sort(teacher_proj, dim=1)
        
        loss = F.mse_loss(student_sorted, teacher_sorted)
        return loss


class AdaptiveUncertaintyMixup:
    """Mixup with curriculum-based uncertainty weighting"""
    def __init__(self, teacher_model, alpha=1.0, num_classes=1000):
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.uncertainty_ema = None
        self.num_classes = num_classes
        self.ema_decay = 0.9
    
    def compute_uncertainty(self, images):
        """Compute prediction uncertainty using entropy and confidence"""
        with torch.no_grad():
            output = self.teacher_model(images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = F.softmax(logits, dim=1)
            
            # Combine entropy and max confidence
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            max_conf = torch.max(probs, dim=1)[0]
            uncertainty = entropy * (1 - max_conf)
            
        return uncertainty
    
    def __call__(self, images, labels, epoch_progress=0.0):
        batch_size = images.size(0)
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(images)
        
        # Update EMA
        if self.uncertainty_ema is None:
            self.uncertainty_ema = uncertainty.mean()
        else:
            self.uncertainty_ema = self.ema_decay * self.uncertainty_ema + (1 - self.ema_decay) * uncertainty.mean()
        
        # Adaptive mixing based on curriculum
        # Early training: more mixing, late training: less mixing
        alpha_schedule = self.alpha * (1.0 - 0.5 * epoch_progress)
        
        indices = torch.randperm(batch_size, device=images.device)
        mixed_images = images.clone()
        
        # Convert hard labels to soft labels if needed
        if len(labels.shape) == 1:
            
            mixed_labels = torch.zeros(batch_size, num_classes, device=labels.device)
            
            for i in range(batch_size):
                # Higher uncertainty = more conservative mixing
                lam = np.random.beta(alpha_schedule, alpha_schedule)
                uncertainty_factor = torch.sigmoid((uncertainty[i] - self.uncertainty_ema) / 0.1).item()
                lam = lam * (1 - 0.3 * uncertainty_factor)
                
                mixed_images[i] = lam * images[i] + (1 - lam) * images[indices[i]]
                
                # Create soft labels
                mixed_labels[i, labels[i]] = lam
                mixed_labels[i, labels[indices[i]]] += (1 - lam)
        else:
            # Already soft labels
            mixed_labels = labels.clone()
            for i in range(batch_size):
                lam = np.random.beta(alpha_schedule, alpha_schedule)
                uncertainty_factor = torch.sigmoid((uncertainty[i] - self.uncertainty_ema) / 0.1).item()
                lam = lam * (1 - 0.3 * uncertainty_factor)
                
                mixed_images[i] = lam * images[i] + (1 - lam) * images[indices[i]]
                mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[indices[i]]
        
        return mixed_images, mixed_labels


class MomentumContrastiveHead(nn.Module):
    """Contrastive head with momentum encoder and hard negative mining"""
    def __init__(self, feature_dim=512, projection_dim=128, momentum=0.999, queue_size=4096):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size
        
        # Query encoder (trainable)
        self.query_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # Key encoder (momentum updated)
        self.key_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # Copy weights
        for param_q, param_k in zip(self.query_proj.parameters(), self.key_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Queue for negative samples
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(self.query_proj.parameters(), self.key_proj.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, query_feat, key_feat, temperature=0.07, use_hard_negatives=True):
        # Pool if needed
        if len(query_feat.shape) == 4:
            query_feat = F.adaptive_avg_pool2d(query_feat, (1, 1)).view(query_feat.size(0), -1)
            key_feat = F.adaptive_avg_pool2d(key_feat, (1, 1)).view(key_feat.size(0), -1)
        
        # Normalize
        q = F.normalize(self.query_proj(query_feat), dim=1)
        
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.key_proj(key_feat), dim=1)
        
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits from queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Hard negative mining
        if use_hard_negatives:
            # Select top-k hardest negatives
            k_hard = min(512, l_neg.size(1))
            hard_neg_logits, _ = torch.topk(l_neg, k_hard, dim=1)
            logits = torch.cat([l_pos, hard_neg_logits], dim=1)
        else:
            logits = torch.cat([l_pos, l_neg], dim=1)
        
        logits /= temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return F.cross_entropy(logits, labels)


class SpectralNormDiscriminator(nn.Module):
    """Multi-scale discriminator with spectral normalization"""
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Spatial discriminator (preserves spatial structure)
        self.spatial_disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(feature_dim, 256, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(64, 1))
        )
        
        # Global discriminator
        self.global_disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, 1))
        )
    
    def forward(self, features):
        if len(features.shape) == 4:
            # Spatial discrimination
            spatial_score = self.spatial_disc(features)
            
            # Global discrimination
            global_feat = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            global_score = self.global_disc(global_feat)
            
            return (spatial_score + global_score) / 2
        else:
            return self.global_disc(features)


class AdaptiveHardSampleMiner:
    """Dynamic hard sample mining with multiple criteria"""
    def __init__(self, initial_percentile=0.7):
        self.percentile = initial_percentile
        self.difficulty_history = []
        self.max_history = 100
    
    def update_percentile(self, epoch_progress):
        """Increase focus on hard samples as training progresses"""
        self.percentile = 0.7 + 0.2 * epoch_progress  # 0.7 -> 0.9
    
    def get_hard_samples(self, student_logits, teacher_logits, labels, student_feat, teacher_feat):
        """Multi-criteria hard sample identification"""
        with torch.no_grad():
            # 1. Prediction disagreement (KL divergence)
            s_probs = F.softmax(student_logits, dim=1)
            t_probs = F.softmax(teacher_logits, dim=1)
            kl_div = torch.sum(t_probs * torch.log(t_probs / (s_probs + 1e-10) + 1e-10), dim=1)
            
            # 2. Prediction confidence
            s_conf = torch.max(s_probs, dim=1)[0]
            confidence_penalty = 1.0 - s_conf
            
            # 3. Feature distance
            s_feat_flat = student_feat.view(student_feat.size(0), -1)
            t_feat_flat = teacher_feat.view(teacher_feat.size(0), -1)
            feat_dist = F.pairwise_distance(
                F.normalize(s_feat_flat, dim=1),
                F.normalize(t_feat_flat, dim=1)
            )
            
            # Combined difficulty score
            difficulty = kl_div + 0.5 * confidence_penalty + 0.3 * feat_dist
            
            # Track difficulty distribution
            self.difficulty_history.append(difficulty.mean().item())
            if len(self.difficulty_history) > self.max_history:
                self.difficulty_history.pop(0)
            
            # Adaptive threshold
            threshold = torch.quantile(difficulty, self.percentile)
            hard_mask = difficulty > threshold
            
        return hard_mask, difficulty


class MSFAMTrainer:
    """Enhanced Multi-Scale Feature Alignment with Adaptive Mixup Trainer"""
    def __init__(self, student_model, teacher_model, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        
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
        self.sliced_wasserstein_loss = SlicedWassersteinLoss(num_projections=50)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Adaptive mixup
        self.mixup = AdaptiveUncertaintyMixup(self.teacher, num_classes=num_classes)
        
        # Momentum contrastive head
        self.contrastive_head = MomentumContrastiveHead(
            feature_dim=512, projection_dim=128, queue_size=4096
        ).to(device)
        
        # Spectral-normalized discriminator
        self.discriminator = SpectralNormDiscriminator(feature_dim=512).to(device)
        
        # Adaptive hard sample miner
        self.hard_miner = AdaptiveHardSampleMiner(initial_percentile=0.7)
        
        self.teacher.eval()
    
    def _get_layer_names(self):
        """Get names of key layers including residuals"""
        layer_names = []
        for name, module in self.student.named_modules():
            # Capture both conv layers and residual connections
            if isinstance(module, nn.Conv2d):
                if any(f'layer{i}' in name for i in range(1, 5)):
                    if name.endswith('.conv2') or 'downsample' in name:
                        layer_names.append(name)
        return layer_names[:10]  # Capture more layers
    
    def compute_feature_alignment_loss(self, images):
        """Compute multi-scale feature alignment with efficient Wasserstein"""
        student_features = self.student_extractor(images)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)
        
        feature_loss = torch.tensor(0.0, device=self.device)
        for layer_name in self.layer_names:
            if layer_name in student_features and layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                # L2 loss (always computed)
                l2_loss = F.mse_loss(s_feat, t_feat)
                
                # Sliced Wasserstein (differentiable, computed every iteration)
                sw_loss = self.sliced_wasserstein_loss(s_feat, t_feat)
                
                feature_loss += l2_loss + 0.1 * sw_loss
        
        return feature_loss / len(self.layer_names)
    
    def compute_adversarial_loss(self, student_features, teacher_features):
        """Compute stable adversarial loss with gradient penalty"""
        # Discriminator predictions
        student_pred = self.discriminator(student_features)
        teacher_pred = self.discriminator(teacher_features.detach())
        
        # Wasserstein GAN loss (more stable)
        student_loss = -student_pred.mean()
        disc_loss = student_pred.detach().mean() - teacher_pred.mean()
        
        # Gradient penalty for Lipschitz constraint
        alpha = torch.rand(student_features.size(0), 1, 1, 1, device=self.device)
        if len(student_features.shape) == 2:
            alpha = alpha.squeeze()
        
        interpolated = (alpha * teacher_features + (1 - alpha) * student_features.detach()).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        disc_loss += 10.0 * gradient_penalty
        
        return student_loss, disc_loss
    
    def train_step(self, images, labels, optimizer, disc_optimizer, epoch, total_epochs, use_mixup=True):
        """Enhanced training step with fixed conceptual issues"""
        self.student.train()
        self.contrastive_head.train()
        self.discriminator.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Epoch progress for curriculum
        epoch_progress = epoch / total_epochs
        
        # Update hard sample miner
        self.hard_miner.update_percentile(epoch_progress)
        
        # Apply curriculum-aware adaptive mixup
        if use_mixup:
            mixed_images, mixed_labels = self.mixup(images, labels, epoch_progress)
        else:
            mixed_images, mixed_labels = images, labels
        
        optimizer.zero_grad()
        disc_optimizer.zero_grad()
        
        # Get features
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        student_logits, student_feat = self.student(mixed_images)
        
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(mixed_images)
        
        # Multi-criteria hard sample mining
        hard_mask, difficulty = self.hard_miner.get_hard_samples(
            student_logits, teacher_logits, labels, student_feat, teacher_feat
        )
        hard_weight = torch.ones_like(difficulty)
        hard_weight[hard_mask] = 2.0
        hard_weight = hard_weight / hard_weight.sum() * len(difficulty)
        
        # Adaptive temperature scheduling
        temperature = 4.0 - 1.0 * epoch_progress  # 4.0 -> 3.0
        
        # Knowledge distillation loss
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # Classification loss with hard sample weighting
        if len(mixed_labels.shape) == 2:
            ce_loss = -torch.mean(
                hard_weight * torch.sum(mixed_labels * F.log_softmax(student_logits, dim=1), dim=1)
            )
        else:
            ce_loss = F.cross_entropy(student_logits, mixed_labels, reduction='none')
            ce_loss = (ce_loss * hard_weight).mean()
        
        # Feature alignment loss
        feat_loss = self.compute_feature_alignment_loss(mixed_images)
        
        # Momentum contrastive loss with hard negatives
        contrastive_temp = 0.07 + 0.03 * (1 - epoch_progress)  # Adaptive temperature
        contrastive_loss = self.contrastive_head(
            student_feat, teacher_feat, temperature=contrastive_temp, use_hard_negatives=True
        )
        
        # Adversarial feature alignment with gradient penalty
        adv_student_loss, adv_disc_loss = self.compute_adversarial_loss(student_feat, teacher_feat)
        
        # Smooth loss weight transition
        lambda_ce = 0.4 - 0.2 * epoch_progress
        lambda_kd = 0.3 + 0.1 * epoch_progress
        lambda_feat = 0.2
        lambda_contrast = 0.05 + 0.1 * epoch_progress
        lambda_adv = 0.05
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_feat * feat_loss +
            lambda_contrast * contrastive_loss +
            lambda_adv * adv_student_loss
        )
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.contrastive_head.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update discriminator (every iteration for WGAN)
        adv_disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        disc_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'feat_loss': feat_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'adv_loss': adv_student_loss.item(),
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
    
    # Optimizer for student and contrastive head
    optimizer = torch.optim.SGD(
        list(student_model.parameters()) + list(trainer.contrastive_head.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Separate optimizer for discriminator (Adam for stability)
    disc_optimizer = torch.optim.Adam(
        trainer.discriminator.parameters(),
        lr=0.0001,
        betas=(0.5, 0.999)
    )
    
    # Cosine annealing with warmup
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
            print(f"Hard sample percentile: {trainer.hard_miner.percentile:.2f}")
    
    trainer.cleanup()
    return student_model
