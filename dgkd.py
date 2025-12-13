"""
Dynamic Graph-based Knowledge Distillation (DGKD)

Novel Contributions:
1. Graph Neural Network for modeling inter-feature relationships
2. Progressive difficulty-aware knowledge transfer
3. Multi-granularity semantic alignment (pixel, patch, channel, spatial)
4. Self-supervised consistency regularization for synthetic data robustness
5. Adaptive temperature scaling based on layer importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """Graph Convolution layer for feature relationship modeling"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
        """
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)
        return output + self.bias


class FeatureGraphBuilder(nn.Module):
    """Build graph structure from feature maps"""
    def __init__(self, num_nodes=64, temperature=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: (batch, channels, h, w)
        Returns:
            nodes: (batch, num_nodes, feature_dim)
            adjacency: (batch, num_nodes, num_nodes)
        """
        batch, channels, h, w = features.shape

        # Adaptive pooling to get fixed number of nodes
        pooled = F.adaptive_avg_pool2d(features, (8, 8))  # (batch, channels, 8, 8)
        nodes = pooled.view(batch, channels, -1).permute(0, 2, 1)  # (batch, 64, channels)

        # Compute adjacency matrix using cosine similarity
        nodes_norm = F.normalize(nodes, p=2, dim=2)
        adjacency = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # (batch, 64, 64)

        # Apply temperature scaling and softmax
        adjacency = F.softmax(adjacency / self.temperature, dim=-1)

        return nodes, adjacency


class MultiGranularityAlignment(nn.Module):
    """Align features at multiple granularity levels"""
    def __init__(self, student_channels, teacher_channels):
        super().__init__()

        # Channel alignment
        self.channel_align = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)

        # Patch-level attention
        self.patch_attention = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(teacher_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(teacher_channels, teacher_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(teacher_channels // 4, teacher_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, student_feat, teacher_feat):
        """
        Returns:
            losses: dict of losses at different granularities
        """
        # Align channels
        student_aligned = self.channel_align(student_feat)

        # Pixel-level loss
        pixel_loss = F.mse_loss(student_aligned, teacher_feat)

        # Patch-level loss with attention
        patch_attn = self.patch_attention(teacher_feat)
        patch_loss = F.mse_loss(student_aligned * patch_attn, teacher_feat * patch_attn)

        # Channel-level loss
        channel_attn = self.channel_attention(teacher_feat)
        channel_loss = F.mse_loss(student_aligned * channel_attn, teacher_feat * channel_attn)

        # Spatial correlation loss
        student_spatial = student_aligned.pow(2).mean(1, keepdim=True)
        teacher_spatial = teacher_feat.pow(2).mean(1, keepdim=True)
        spatial_loss = F.mse_loss(student_spatial, teacher_spatial)

        return {
            'pixel': pixel_loss,
            'patch': patch_loss,
            'channel': channel_loss,
            'spatial': spatial_loss
        }


class ProgressiveKnowledgeTransfer(nn.Module):
    """Transfer knowledge progressively based on difficulty"""
    def __init__(self, num_epochs=3000, warmup_epochs=500):
        super().__init__()
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_weights(self):
        """Get dynamic weights based on training progress"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup: focus on easy features (lower layers)
            progress = self.current_epoch / self.warmup_epochs
            layer_weights = [1.0, progress * 0.8, progress * 0.6, progress * 0.4]
        else:
            # Progressive: gradually focus on harder features (higher layers)
            progress = (self.current_epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            layer_weights = [0.5, 0.7 + progress * 0.3, 0.9 + progress * 0.1, 1.0]

        return layer_weights


class SelfSupervisedConsistency(nn.Module):
    """Self-supervised consistency for synthetic data robustness"""
    def __init__(self):
        super().__init__()

    def forward(self, student_outputs, augmented_outputs):
        """
        Enforce consistency between original and augmented inputs
        Args:
            student_outputs: dict with 'features' and 'logits'
            augmented_outputs: dict with 'features' and 'logits' from augmented input
        """
        # Logit consistency
        logit_consistency = F.kl_div(
            F.log_softmax(student_outputs['logits'], dim=1),
            F.softmax(augmented_outputs['logits'].detach(), dim=1),
            reduction='batchmean'
        )

        # Feature consistency
        feature_consistency = 0
        if student_outputs['features'] is not None and augmented_outputs['features'] is not None:
            for s_feat, a_feat in zip(student_outputs['features'], augmented_outputs['features']):
                feature_consistency += F.mse_loss(s_feat, a_feat.detach())
            feature_consistency /= len(student_outputs['features'])

        return logit_consistency + 0.5 * feature_consistency


class DGKDLoss(nn.Module):
    """Dynamic Graph-based Knowledge Distillation Loss"""

    def __init__(self, teacher_channels, student_channels, num_epochs=3000):
        super().__init__()

        self.num_layers = len(teacher_channels)

        # Graph components for each layer
        self.graph_builders = nn.ModuleList([
            FeatureGraphBuilder(num_nodes=64) for _ in range(self.num_layers)
        ])

        self.graph_convs = nn.ModuleList([
            GraphConvolution(t_ch, t_ch) for t_ch in teacher_channels
        ])

        # Multi-granularity alignment
        self.mg_alignments = nn.ModuleList([
            MultiGranularityAlignment(s_ch, t_ch)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

        # Progressive transfer
        self.progressive = ProgressiveKnowledgeTransfer(num_epochs=num_epochs)

        # Self-supervised consistency
        self.consistency = SelfSupervisedConsistency()

        # Adaptive temperature for KD
        self.base_temperature = 4.0

    def graph_distillation_loss(self, student_feat, teacher_feat, layer_idx):
        """Graph-based relational distillation"""
        # Build graphs
        s_nodes, s_adj = self.graph_builders[layer_idx](student_feat)
        t_nodes, t_adj = self.graph_builders[layer_idx](teacher_feat)

        # Apply graph convolution
        s_graph_feat = self.graph_convs[layer_idx](s_nodes, s_adj)
        t_graph_feat = self.graph_convs[layer_idx](t_nodes, t_adj)

        # Graph feature loss
        graph_feat_loss = F.mse_loss(s_graph_feat, t_graph_feat.detach())

        # Graph structure loss (adjacency matrix)
        graph_struct_loss = F.kl_div(
            F.log_softmax(s_adj.view(-1, s_adj.size(-1)), dim=1),
            F.softmax(t_adj.view(-1, t_adj.size(-1)), dim=1),
            reduction='batchmean'
        )

        return graph_feat_loss + 0.5 * graph_struct_loss

    def forward(self, student_outputs, teacher_outputs, labels, epoch=None, augmented_outputs=None):
        """
        Args:
            student_outputs: dict with 'features' (list) and 'logits'
            teacher_outputs: dict with 'features' (list) and 'logits'
            labels: ground truth labels
            epoch: current training epoch
            augmented_outputs: optional augmented student outputs for consistency
        """
        if epoch is not None:
            self.progressive.set_epoch(epoch)

        # Get progressive weights
        layer_weights = self.progressive.get_weights()

        # Feature distillation with graph and multi-granularity
        feature_loss = 0
        graph_loss = 0
        mg_losses = {'pixel': 0, 'patch': 0, 'channel': 0, 'spatial': 0}

        for i, (s_feat, t_feat) in enumerate(zip(
            student_outputs['features'],
            teacher_outputs['features']
        )):
            weight = layer_weights[i]

            # Graph-based relational loss
            graph_loss += weight * self.graph_distillation_loss(s_feat, t_feat, i)

            # Multi-granularity alignment
            mg_loss_dict = self.mg_alignments[i](s_feat, t_feat)
            for key in mg_losses:
                mg_losses[key] += weight * mg_loss_dict[key]

        # Average losses
        graph_loss = graph_loss / self.num_layers
        for key in mg_losses:
            mg_losses[key] = mg_losses[key] / self.num_layers

        # Combine multi-granularity losses
        mg_total = mg_losses['pixel'] + 0.8 * mg_losses['patch'] + 0.6 * mg_losses['channel'] + 0.5 * mg_losses['spatial']

        # Adaptive temperature KD on logits
        progress = self.progressive.current_epoch / self.progressive.num_epochs
        temperature = self.base_temperature * (1.0 - 0.5 * progress)  # Decrease temperature over time

        kd_loss = F.kl_div(
            F.log_softmax(student_outputs['logits'] / temperature, dim=1),
            F.softmax(teacher_outputs['logits'] / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Classification loss
        cls_loss = F.cross_entropy(student_outputs['logits'], labels)

        # Self-supervised consistency (if augmented data provided)
        consistency_loss = 0
        if augmented_outputs is not None:
            consistency_loss = self.consistency(student_outputs, augmented_outputs)

        # Combine all losses with balanced weights
        total_loss = (
            0.3 * cls_loss +           # Classification
            1.5 * kd_loss +             # Knowledge distillation (main)
            0.8 * graph_loss +          # Graph relational
            0.6 * mg_total +            # Multi-granularity
            0.3 * consistency_loss      # Self-supervised consistency
        )

        return {
            'total': total_loss,
            'cls_loss': cls_loss.item(),
            'kd_loss': kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
            'graph_loss': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else graph_loss,
            'mg_loss': mg_total.item() if isinstance(mg_total, torch.Tensor) else mg_total,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'temperature': temperature
        }


def extract_features_for_dgkd(model, images, layer_names):
    """
    Extract features from multiple layers for DGKD

    Args:
        model: Neural network model
        images: Input images
        layer_names: List of layer names to extract features from

    Returns:
        Dict with 'features' (list of tensors) and 'logits'
    """
    features = []
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            features.append(output.clone())
        return hook

    # Register hooks
    for name in layer_names:
        layer = dict(model.named_modules()).get(name)
        if layer is not None:
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # Forward pass
    with torch.no_grad():
        output = model(images)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Handle output format
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    return {
        'features': features,
        'logits': logits
    }