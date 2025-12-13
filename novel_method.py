
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadFeatureAttention(nn.Module):
    """Multi-head attention for feature alignment"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, student_feat, teacher_feat):
        bs = student_feat.size(0)
        s_flat = student_feat.view(bs, -1)
        t_flat = teacher_feat.view(bs, -1)
        
        # Project to query, key, value
        q = self.q_proj(s_flat).view(bs, self.num_heads, self.head_dim)
        k = self.k_proj(t_flat).view(bs, self.num_heads, self.head_dim)
        v = self.v_proj(t_flat).view(bs, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.view(bs, -1)
        out = self.out_proj(out)
        
        return F.mse_loss(out, t_flat)


class FeatureDiscriminator(nn.Module):
    """Discriminator for adversarial feature matching"""
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 4, 1)
        )
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)


class ContrastiveFeatureLoss(nn.Module):
    """Contrastive loss for feature alignment"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_feat, teacher_feat):
        bs = student_feat.size(0)
        
        # Flatten and normalize
        s_flat = F.normalize(student_feat.view(bs, -1), dim=1)
        t_flat = F.normalize(teacher_feat.view(bs, -1), dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(s_flat, t_flat.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(bs, device=student_feat.device)
        
        # Contrastive loss (symmetric)
        loss_s = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        
        return (loss_s + loss_t) / 2


class SpatialAttentionMap(nn.Module):
    """Generate spatial attention maps from features"""
    def __init__(self):
        super().__init__()
        
    def forward(self, features):
        # Channel-wise attention
        channel_attn = torch.mean(features, dim=1, keepdim=True)
        
        # Spatial attention
        max_pool = torch.max(features, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        spatial_attn = torch.cat([max_pool, avg_pool], dim=1)
        
        return channel_attn, spatial_attn


class RobustFeatureExtractor(nn.Module):
    """Extract features from multiple layers"""
    def __init__(self, model, rm_blocks):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        self.rm_blocks = set(rm_blocks) if rm_blocks else set()
        
        # Find valid layers
        target_layers = self._find_valid_layers()
        
        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self.save_feature(name))
                self.hooks.append(hook)
        
        print(f"Extracting features from {len(target_layers)} layers: {target_layers}")
    
    def _find_valid_layers(self):
        valid_layers = []
        for name, module in self.model.named_modules():
            if any(rb in name for rb in self.rm_blocks):
                continue
            if 'conv2' in name or 'conv3' in name:
                if isinstance(module, nn.Conv2d):
                    valid_layers.append(name)
        return valid_layers[:8]
    
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


class EnhancedMSFAMTrainer:
    """Enhanced trainer with attention, contrastive learning, and adversarial matching"""
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Feature extractors
        self.student_extractor = RobustFeatureExtractor(self.student, rm_blocks)
        self.teacher_extractor = RobustFeatureExtractor(self.teacher, [])
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            self.student.get_feat = 'pre_GAP'
            _, sample_feat = self.student(dummy_input)
            feature_dim = sample_feat.view(sample_feat.size(0), -1).size(1)
        
        # Advanced loss modules
        self.attention_loss = MultiHeadFeatureAttention(feature_dim, num_heads=8).to(device)
        self.contrastive_loss = ContrastiveFeatureLoss(temperature=0.07)
        self.spatial_attn = SpatialAttentionMap()
        
        # Adversarial components
        self.discriminator = FeatureDiscriminator(feature_dim).to(device)
        self.disc_optimizer = None  # Will be set in train_step
        
        # Standard losses
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        
        self.teacher.eval()
        
        # Curriculum tracking
        self.sample_difficulties = {}
        self.epoch_counter = 0
        
    def compute_adversarial_loss(self, student_feat, teacher_feat):
        """Adversarial feature matching"""
        # Create detached copies for discriminator training
        student_feat_detached = student_feat.detach()
        teacher_feat_detached = teacher_feat.detach()
        
        # Discriminator loss (separate from main graph)
        real_score = self.discriminator(teacher_feat_detached)
        fake_score_d = self.discriminator(student_feat_detached)
        
        disc_loss = torch.mean(fake_score_d) - torch.mean(real_score)
        
        # Gradient penalty for stability
        alpha = torch.rand(student_feat.size(0), 1, 1, 1, device=self.device)
        interpolated = (alpha * teacher_feat_detached + (1 - alpha) * student_feat_detached).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        disc_loss = disc_loss + 10 * gradient_penalty
        
        # Generator loss (connected to main graph, no detach)
        fake_score_g = self.discriminator(student_feat)
        gen_loss = -torch.mean(fake_score_g)
        
        return gen_loss, disc_loss
    
    def compute_multi_scale_feature_loss(self, images):
        """Multi-scale feature alignment"""
        student_features = self.student_extractor(images)
        
        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)
        
        total_loss = 0.0
        count = 0
        
        for layer_name in student_features.keys():
            if layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                if s_feat.shape != t_feat.shape:
                    continue
                
                # Multi-component feature loss
                mse_loss = F.mse_loss(s_feat, t_feat)
                
                # Spatial attention matching
                s_ch_attn, s_sp_attn = self.spatial_attn(s_feat)
                t_ch_attn, t_sp_attn = self.spatial_attn(t_feat)
                attn_loss = F.mse_loss(s_ch_attn, t_ch_attn) + F.mse_loss(s_sp_attn, t_sp_attn)
                
                total_loss += mse_loss + 0.5 * attn_loss
                count += 1
        
        return total_loss / max(count, 1) if count > 0 else torch.tensor(0.0).to(self.device)
    
    def estimate_sample_difficulty(self, student_logits, labels):
        """Estimate difficulty of samples for curriculum learning"""
        with torch.no_grad():
            probs = F.softmax(student_logits, dim=1)
            if len(labels.shape) == 1:
                # Hard labels
                correct_probs = probs[torch.arange(len(labels)), labels]
                difficulty = 1.0 - correct_probs
            else:
                # Soft labels
                label_entropy = -torch.sum(labels * torch.log(labels + 1e-8), dim=1)
                pred_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                difficulty = torch.abs(label_entropy - pred_entropy)
        
        return difficulty
    
    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Enhanced training step with multiple loss components"""
        self.student.train()
        self.epoch_counter = epoch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Initialize discriminator optimizer if not exists
        if self.disc_optimizer is None:
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=0.0001,
                betas=(0.5, 0.999)
            )
        
        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'
        
        # Forward pass
        student_logits, student_feat = self.student(images)
        
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)
        
        # 1. Classification loss
        if len(labels.shape) == 1:
            ce_loss = self.ce_loss(student_logits, labels)
        else:
            ce_loss = -torch.mean(torch.sum(labels * F.log_softmax(student_logits, dim=1), dim=1))
        
        # 2. KD loss with adaptive temperature
        base_temp = 4.0
        progress = epoch / total_epochs
        temperature = base_temp * (1.0 - 0.5 * progress)  # Decrease temperature over time
        
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # 3. Multi-head attention loss
        attn_loss = self.attention_loss(student_feat, teacher_feat)
        
        # 4. Contrastive feature loss
        contrastive_loss = self.contrastive_loss(student_feat, teacher_feat)
        
        # 5. Multi-scale feature alignment
        if epoch % 2 == 0:
            multi_scale_loss = self.compute_multi_scale_feature_loss(images)
        else:
            multi_scale_loss = torch.tensor(0.0).to(self.device)
        
        # 6. Adversarial feature matching
        gen_loss, disc_loss = self.compute_adversarial_loss(student_feat, teacher_feat)
        
        # Update discriminator (separate from main optimization)
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        
        # 7. Feature magnitude matching
        s_feat_norm = torch.norm(student_feat.view(student_feat.size(0), -1), dim=1)
        t_feat_norm = torch.norm(teacher_feat.view(teacher_feat.size(0), -1), dim=1)
        norm_loss = F.mse_loss(s_feat_norm, t_feat_norm)
        
        # Progressive loss weights
        if epoch < 200:
            # Early phase: focus on basic alignment
            lambda_ce = 0.2
            lambda_kd = 0.6
            lambda_attn = 0.1
            lambda_contrast = 0.05
            lambda_multi = 0.05
            lambda_adv = 0.0
            lambda_norm = 0.0
        elif epoch < 1000:
            # Middle phase: introduce advanced losses
            alpha = (epoch - 200) / 800
            lambda_ce = 0.15
            lambda_kd = 0.5
            lambda_attn = 0.15
            lambda_contrast = 0.1
            lambda_multi = 0.1 * alpha
            lambda_adv = 0.05 * alpha
            lambda_norm = 0.05 * alpha
        else:
            # Late phase: balanced training
            lambda_ce = 0.1
            lambda_kd = 0.4
            lambda_attn = 0.2
            lambda_contrast = 0.15
            lambda_multi = 0.1
            lambda_adv = 0.05
            lambda_norm = 0.05
        
        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_attn * attn_loss +
            lambda_contrast * contrastive_loss +
            lambda_multi * multi_scale_loss +
            lambda_adv * gen_loss +
            lambda_norm * norm_loss
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'attn_loss': attn_loss.item(),
            'contrast_loss': contrastive_loss.item(),
            'multi_scale_loss': multi_scale_loss.item(),
            'adv_loss': gen_loss.item(),
            'norm_loss': norm_loss.item(),
            'disc_loss': disc_loss.item()
        }
    
    def cleanup(self):
        """Remove hooks"""
        self.student_extractor.remove_hooks()
        self.teacher_extractor.remove_hooks()


# Backward compatibility
SimplifiedMSFAMTrainer = EnhancedMSFAMTrainer
MSFAMTrainer = EnhancedMSFAMTrainer


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000):
    """Train student model using enhanced MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = EnhancedMSFAMTrainer(student_model, teacher_model, num_classes=num_classes, device=device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.SGD(
        list(student_model.parameters()) + list(trainer.attention_loss.parameters()),
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
            'attn_loss': 0,
            'contrast_loss': 0,
            'multi_scale_loss': 0,
            'adv_loss': 0,
            'norm_loss': 0,
            'disc_loss': 0
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
            print(f"  Total: {epoch_losses['total_loss']:.4f} | CE: {epoch_losses['ce_loss']:.4f} | KD: {epoch_losses['kd_loss']:.4f}")
            print(f"  Attn: {epoch_losses['attn_loss']:.4f} | Contrast: {epoch_losses['contrast_loss']:.4f}")
            print(f"  Multi: {epoch_losses['multi_scale_loss']:.4f} | Adv: {epoch_losses['adv_loss']:.4f} | Norm: {epoch_losses['norm_loss']:.4f}")
            print(f"  Disc: {epoch_losses['disc_loss']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    trainer.cleanup()
    return student_model


# Keep simplified version as backup
train_with_simplified_msfam = train_with_msfam
