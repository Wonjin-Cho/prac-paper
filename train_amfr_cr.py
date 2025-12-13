
"""
Training script using the novel AMFR-CR method
"""

import torch
import torch.nn as nn
from amfr_cr import AMFRCRLoss, extract_multi_scale_features
from util import AverageMeter, accuracy


def train_with_amfr_cr(
    train_loader,
    student_model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    args,
    layer_names_student,
    layer_names_teacher
):
    """
    Train one epoch using AMFR-CR method
    
    Args:
        train_loader: DataLoader for training data
        student_model: Pruned student model
        teacher_model: Original teacher model
        criterion: AMFRCRLoss instance
        optimizer: Optimizer
        epoch: Current epoch
        args: Training arguments
        layer_names_student: List of student layer names for feature extraction
        layer_names_teacher: List of teacher layer names for feature extraction
    """
    losses = AverageMeter()
    ms_losses = AverageMeter()
    contrast_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Switch to train mode
    student_model.train()
    teacher_model.eval()
    
    for i, (images, labels) in enumerate(train_loader):
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()
        
        # Extract multi-scale features from teacher
        with torch.no_grad():
            teacher_outputs = extract_multi_scale_features(
                teacher_model, images, layer_names_teacher
            )
        
        # Extract multi-scale features from student
        student_outputs = extract_multi_scale_features(
            student_model, images, layer_names_student
        )
        
        # Compute AMFR-CR loss
        loss_dict = criterion(student_outputs, teacher_outputs, labels, images)
        
        # Update metrics
        losses.update(loss_dict['total'].item(), images.size(0))
        ms_losses.update(loss_dict['ms_loss'], images.size(0))
        contrast_losses.update(loss_dict['contrast_loss'], images.size(0))
        
        # Measure accuracy
        prec1, prec5 = accuracy(student_outputs['logits'], labels, topk=(1, 5))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss_dict['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'MS-Loss {ms_losses.val:.4f} ({ms_losses.avg:.4f})\t'
                  f'Contrast {contrast_losses.val:.4f} ({contrast_losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    return losses.avg, top1.avg


def setup_amfr_cr_training(student_model, teacher_model, args):
    """
    Setup AMFR-CR training components
    
    Returns:
        criterion, optimizer, layer_names
    """
    # Define layer names for feature extraction
    # Adjust based on your ResNet architecture
    layer_names_student = ['layer1', 'layer2', 'layer3']
    layer_names_teacher = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # Get channel dimensions
    student_channels = [64, 128, 256]  # For ResNet34 after pruning
    teacher_channels = [64, 128, 256, 512]  # For ResNet34 original
    feature_dim = 512  # Final feature dimension
    
    # Create AMFR-CR loss
    criterion = AMFRCRLoss(
        teacher_channels[:3],  # Match student layers
        student_channels,
        feature_dim
    )
    
    if args.cuda:
        criterion = criterion.cuda()
    
    # Optimizer with different learning rates for different components
    params = [
        {'params': student_model.parameters(), 'lr': args.lr},
        {'params': criterion.parameters(), 'lr': args.lr * 0.1}
    ]
    
    optimizer = torch.optim.SGD(
        params,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.001
    )
    
    return criterion, optimizer, scheduler, layer_names_student, layer_names_teacher
