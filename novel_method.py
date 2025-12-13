import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveFeatureAligner(nn.Module):
    """Lightweight feature aligner that adapts to channel mismatches"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.adapter = None

    def forward(self, x):
        if self.adapter is not None:
            return self.adapter(x)
        return x


class RobustFeatureExtractor(nn.Module):
    """Extract features only from layers that exist in the pruned model"""
    def __init__(self, model, rm_blocks):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        self.rm_blocks = set(rm_blocks) if rm_blocks else set()

        # Dynamically find available layers
        target_layers = self._find_valid_layers()

        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self.save_feature(name))
                self.hooks.append(hook)

        print(f"Extracting features from {len(target_layers)} layers: {target_layers}")

    def _find_valid_layers(self):
        """Find layers that exist in the pruned model"""
        valid_layers = []
        for name, module in self.model.named_modules():
            # Skip removed blocks
            if any(rb in name for rb in self.rm_blocks):
                continue
            # Target final conv in each remaining block
            if 'conv2' in name or 'conv3' in name:
                if isinstance(module, nn.Conv2d):
                    valid_layers.append(name)
        return valid_layers[:8]  # Limit to 8 layers to avoid overhead

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


class EfficientFeatureAlignment(nn.Module):
    """Efficient feature alignment using cosine similarity instead of Wasserstein"""
    def __init__(self):
        super().__init__()

    def forward(self, student_feat, teacher_feat):
        # Flatten features
        bs = student_feat.size(0)
        s_flat = student_feat.view(bs, -1)
        t_flat = teacher_feat.view(bs, -1)

        # L2 normalize
        s_norm = F.normalize(s_flat, p=2, dim=1)
        t_norm = F.normalize(t_flat, p=2, dim=1)

        # Cosine similarity loss (1 - similarity)
        cosine_sim = torch.sum(s_norm * t_norm, dim=1)
        loss = torch.mean(1.0 - cosine_sim)

        # Add MSE component for magnitude matching
        mse_loss = F.mse_loss(s_flat, t_flat)

        return loss + 0.1 * mse_loss


class SimplifiedMSFAMTrainer:
    """Simplified MSFAM without mixup on synthetic data"""
    def __init__(self, student_model, teacher_model, rm_blocks=None, num_classes=1000, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.num_classes = num_classes

        # Feature extractors that handle pruned architectures
        self.student_extractor = RobustFeatureExtractor(self.student, rm_blocks)
        self.teacher_extractor = RobustFeatureExtractor(self.teacher, [])

        # Loss functions
        self.feature_align = EfficientFeatureAlignment()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.teacher.eval()

        # Progressive loss scheduling
        self.loss_schedule = {
            'warmup_iters': 200,
            'feature_rampup_iters': 1000
        }

    def compute_multi_scale_feature_loss(self, images):
        """Compute feature alignment at multiple scales"""
        student_features = self.student_extractor(images)

        with torch.no_grad():
            teacher_features = self.teacher_extractor(images)

        total_loss = 0.0
        count = 0

        # Match features from common layers
        for layer_name in student_features.keys():
            if layer_name in teacher_features:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]

                # Skip if shapes don't match (shouldn't happen with robust extractor)
                if s_feat.shape != t_feat.shape:
                    continue

                loss = self.feature_align(s_feat, t_feat)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1) if count > 0 else torch.tensor(0.0).to(self.device)

    def train_step(self, images, labels, optimizer, epoch, total_epochs):
        """Training step without mixup - direct knowledge distillation"""
        self.student.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()

        # Progress for loss scheduling
        progress = min(epoch / total_epochs, 1.0)

        # Set feature extraction mode
        self.student.get_feat = 'pre_GAP'
        self.teacher.get_feat = 'pre_GAP'

        # Forward pass (NO MIXUP - use original synthetic data)
        student_logits, student_feat = self.student(images)

        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(images)

        # 1. KD loss with temperature scaling
        temperature = 4.0  # Keep constant for synthetic data
        kd_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        # 2. Classification loss (hard labels if available, else use teacher's prediction)
        if len(labels.shape) == 1:
            ce_loss = self.ce_loss(student_logits, labels)
        else:
            # Soft labels - use cross entropy with soft targets
            ce_loss = -torch.mean(torch.sum(labels * F.log_softmax(student_logits, dim=1), dim=1))

        # 3. Final layer feature alignment
        final_feat_loss = self.feature_align(student_feat, teacher_feat)

        # 4. Multi-scale feature alignment (computed less frequently to save time)
        if epoch % 2 == 0:  # Every other iteration
            multi_scale_loss = self.compute_multi_scale_feature_loss(images)
        else:
            multi_scale_loss = torch.tensor(0.0).to(self.device)

        # Progressive loss weights - start with KD, gradually add feature alignment
        if epoch < self.loss_schedule['warmup_iters']:
            # Warmup: focus on KD and CE
            lambda_ce = 0.3
            lambda_kd = 0.7
            lambda_final = 0.0
            lambda_multi = 0.0
        elif epoch < self.loss_schedule['feature_rampup_iters']:
            # Rampup: gradually introduce feature alignment
            alpha = (epoch - self.loss_schedule['warmup_iters']) / \
                    (self.loss_schedule['feature_rampup_iters'] - self.loss_schedule['warmup_iters'])
            lambda_ce = 0.2
            lambda_kd = 0.5
            lambda_final = 0.2 * alpha
            lambda_multi = 0.1 * alpha
        else:
            # Full training: balanced losses
            lambda_ce = 0.15
            lambda_kd = 0.45
            lambda_final = 0.25
            lambda_multi = 0.15

        # Combined loss
        total_loss = (
            lambda_ce * ce_loss +
            lambda_kd * kd_loss +
            lambda_final * final_feat_loss +
            lambda_multi * multi_scale_loss
        )

        # Backward and optimize
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)

        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'feat_loss': final_feat_loss.item(),
            'final_feat_loss': multi_scale_loss.item()
        }

    def cleanup(self):
        """Remove hooks"""
        self.student_extractor.remove_hooks()
        self.teacher_extractor.remove_hooks()


def train_with_msfam(student_model, teacher_model, train_loader, epochs=2000, lr=0.01, num_classes=1000):
    """Train student model using simplified MSFAM method"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = SimplifiedMSFAMTrainer(student_model, teacher_model, num_classes, device)

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
            print(f"  Multi-Scale Loss: {epoch_losses['final_feat_loss']:.4f}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    trainer.cleanup()
    return student_model


# Backward compatibility
MSFAMTrainer = SimplifiedMSFAMTrainer
train_with_simplified_msfam = train_with_msfam