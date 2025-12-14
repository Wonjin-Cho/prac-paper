
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyFilter:
    """Apply frequency domain filtering to images"""
    
    @staticmethod
    def low_pass_filter(images, cutoff_ratio=0.5):
        """
        Apply low-pass filter in frequency domain
        Args:
            images: [B, C, H, W] tensor
            cutoff_ratio: ratio of frequencies to keep (0-1)
        """
        B, C, H, W = images.shape
        
        # Apply FFT
        fft_images = torch.fft.fft2(images)
        fft_shifted = torch.fft.fftshift(fft_images)
        
        # Create low-pass mask
        crow, ccol = H // 2, W // 2
        radius = int(min(H, W) * cutoff_ratio / 2)
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.to(images.device), x.to(images.device)
        mask = ((y - crow) ** 2 + (x - ccol) ** 2) <= radius ** 2
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        fft_filtered = fft_shifted * mask
        
        # Inverse FFT
        fft_ishifted = torch.fft.ifftshift(fft_filtered)
        filtered_images = torch.fft.ifft2(fft_ishifted).real
        
        return filtered_images
    
    @staticmethod
    def high_pass_filter(images, cutoff_ratio=0.1):
        """Apply high-pass filter to enhance edges"""
        low_freq = FrequencyFilter.low_pass_filter(images, cutoff_ratio)
        high_freq = images - low_freq
        return high_freq
    
    @staticmethod
    def adaptive_frequency_mix(images, teacher_features, alpha=0.7):
        """
        Mix low and high frequency components adaptively based on feature confidence
        Args:
            images: input images
            teacher_features: teacher model features for confidence estimation
            alpha: mixing ratio (higher = more low frequency)
        """
        low_freq = FrequencyFilter.low_pass_filter(images, cutoff_ratio=0.6)
        high_freq = FrequencyFilter.high_pass_filter(images, cutoff_ratio=0.15)
        
        # Adaptive mixing based on feature variance (confidence proxy)
        feature_var = torch.var(teacher_features, dim=(2, 3), keepdim=True)
        feature_var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
        
        # High variance -> use more original, low variance -> use more filtered
        adaptive_alpha = alpha * (1 - feature_var_norm.mean(dim=1, keepdim=True))
        
        mixed = adaptive_alpha * low_freq + (1 - adaptive_alpha) * images
        return mixed


class AdaptiveNoiseInjection(nn.Module):
    """Inject adaptive noise to make training more robust"""
    
    def __init__(self, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, images, difficulty_score=None):
        """
        Add noise proportional to difficulty
        difficulty_score: [B] tensor, higher = more difficult samples
        """
        if difficulty_score is None:
            difficulty_score = torch.ones(images.size(0), device=images.device)
        
        # Normalize difficulty score
        difficulty_score = (difficulty_score - difficulty_score.min()) / (difficulty_score.max() - difficulty_score.min() + 1e-8)
        
        # Generate adaptive noise
        noise = torch.randn_like(images) * self.noise_std
        noise = noise * difficulty_score.view(-1, 1, 1, 1)
        
        return images + noise


class SpectralNormalization:
    """Normalize images in frequency domain"""
    
    @staticmethod
    def spectral_normalize(images):
        """Normalize spectral distribution to match natural images"""
        # Apply FFT
        fft_images = torch.fft.fft2(images)
        fft_shifted = torch.fft.fftshift(fft_images)
        
        # Compute magnitude and phase
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        
        # Normalize magnitude to have natural 1/f decay
        H, W = images.shape[-2:]
        crow, ccol = H // 2, W // 2
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.to(images.device), x.to(images.device)
        
        # Distance from center
        distance = torch.sqrt((y - crow) ** 2 + (x - ccol) ** 2) + 1
        
        # Apply 1/f normalization
        target_magnitude = 1.0 / distance.unsqueeze(0).unsqueeze(0)
        target_magnitude = target_magnitude / target_magnitude.max()
        
        # Blend with original magnitude
        normalized_magnitude = 0.7 * magnitude + 0.3 * (magnitude.mean() * target_magnitude)
        
        # Reconstruct
        fft_normalized = normalized_magnitude * torch.exp(1j * phase)
        fft_ishifted = torch.fft.ifftshift(fft_normalized)
        normalized_images = torch.fft.ifft2(fft_ishifted).real
        
        return normalized_images


def compute_difficulty_score(student_logits, teacher_logits, labels):
    """
    Compute per-sample difficulty score based on prediction disagreement
    Returns: [B] tensor with difficulty scores
    """
    # Softmax probabilities
    s_probs = F.softmax(student_logits, dim=1)
    t_probs = F.softmax(teacher_logits, dim=1)
    
    # KL divergence per sample
    kl_div = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        t_probs,
        reduction='none'
    ).sum(dim=1)
    
    # Prediction confidence difference
    s_conf, _ = s_probs.max(dim=1)
    t_conf, _ = t_probs.max(dim=1)
    conf_diff = torch.abs(s_conf - t_conf)
    
    # Combined difficulty score
    difficulty = kl_div + conf_diff
    
    return difficulty
