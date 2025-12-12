from __future__ import print_function, division

import sys
import time
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from MIXSTD import MIXSTDLoss
from torch.utils.data import Dataset, DataLoader

from util import AverageMeter, accuracy

## Partial Mixup
def PMU(inputs, targets, percent, beta_a, mixup=True, num_classes=100, problematic_classes=None):
    batch_size = inputs.shape[0]
    img_size = inputs.shape[2]  # Get image size dynamically

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    if problematic_classes is not None:
        # Convert to a set for fast lookup
        problematic_classes_set = set(problematic_classes)
        
        # Get indices of samples where label is in problematic_classes
        mask = torch.tensor([label.item() in problematic_classes_set for label in targets], device=targets.device)
        problem_indices = torch.where(mask)[0]

        if len(problem_indices) == 0:
            raise ValueError("No samples from problematic classes found in this batch.")
        
        # Sample from problem_indices to get rp2
        if len(problem_indices) < batch_size:
            # Sample with replacement
            rp2 = problem_indices[torch.randint(0, len(problem_indices), (batch_size,), device=targets.device)]
        else:
            # Sample without replacement
            rp2 = problem_indices[torch.randperm(len(problem_indices))[:batch_size]]

        inputs2 = inputs[rp2]
        targets2 = targets[rp2]
        targets2_1 = targets2.unsqueeze(1)

    else:
        rp2 = torch.randperm(batch_size)
        inputs2 = inputs[rp2]
        targets2 = targets[rp2]
        targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, num_classes)
    y_onehot.zero_()
    targets1_1 = targets1_1.cpu()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)
    targets1_oh = targets1_oh.cuda()

    y_onehot2 = torch.FloatTensor(batch_size, num_classes)
    y_onehot2.zero_()
    targets2_1 = targets2_1.cpu()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)
    targets2_oh = targets2_oh.cuda()

    if mixup is True:
        a = numpy.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))
    
    ridx = torch.randint(0,batch_size,(int(a.shape[0] * percent),))
    a[ridx] = 1.

    b = numpy.tile(a[..., None, None], [1, 3, img_size, img_size])

    # b = b.cuda()
    inputs1 = inputs1.cuda()
    inputs2 = inputs2.cuda()
    inputs1 = inputs1 * torch.from_numpy(b).float().cuda()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float().cuda()
    # inputs1 = inputs1.cuda()
    # inputs2 = inputs2.cuda()

    c = numpy.tile(a, [1, num_classes])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float().cuda()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float().cuda()

    inputs_shuffle = inputs1 + inputs2
    inputs_shuffle = inputs_shuffle.cuda()
    targets_shuffle = targets1_oh + targets2_oh
    targets_shuffle = targets_shuffle.cuda()

    return inputs_shuffle, targets_shuffle


## Full Mixup
def FMU(inputs, targets, beta_a, teacher_model, mixup=True, num_classes=100, problematic_classes = None):
    batch_size = inputs.shape[0]
    img_size = inputs.shape[2]  # Get image size dynamically

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)
    problematic_classes = None
    if problematic_classes is not None:
        # Convert to a set for fast lookup
        problematic_classes_set = set(problematic_classes)
        
        # Get indices of samples where label is in problematic_classes
        mask = torch.tensor([label.item() in problematic_classes_set for label in targets], device=targets.device)
        problem_indices = torch.where(mask)[0]

        if len(problem_indices) == 0:
            raise ValueError("No samples from problematic classes found in this batch.")
        
        # Sample from problem_indices to get rp2
        if len(problem_indices) < batch_size:
            # Sample with replacement
            rp2 = problem_indices[torch.randint(0, len(problem_indices), (batch_size,), device=targets.device)]
        else:
            # Sample without replacement
            rp2 = problem_indices[torch.randperm(len(problem_indices))[:batch_size]]

        inputs2 = inputs[rp2]
        targets2 = targets[rp2]
        targets2_1 = targets2.unsqueeze(1)

    else:
        rp2 = torch.randperm(batch_size)
        inputs2 = inputs[rp2]
        targets2 = targets[rp2]
        targets2_1 = targets2.unsqueeze(1)

    if mixup is True:
        a = numpy.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))
    mixing_weights = torch.from_numpy(a).float().to(inputs.device)
    
    b = numpy.tile(a[..., None, None], [1, 3, img_size, img_size])

    # Convert to tensors and move to the same device as inputs
    b_tensor = torch.from_numpy(b).float().to(inputs.device)
    
    inputs1 = inputs1 * b_tensor
    inputs2 = inputs2 * torch.from_numpy(1 - b).float().to(inputs.device)

    # Create the combined/mixed input
    inputs_shuffle = inputs1 + inputs2

    # Get teacher logits for the combined input
    with torch.no_grad():
        inputs_shuffle = inputs_shuffle.cuda()
        teacher_output = teacher_model(inputs_shuffle)
        # for i in range(len(a)):
        #     inputs1_dominates = a[i][0] > 0.5
        #     if inputs1_dominates:
        #         teacher_output[i] = targets1[i]
        #     else: 
        #         teacher_output[i] = targets2[i]
        pseudo_labels = torch.where(
            mixing_weights.squeeze(1) > 0.5,
            targets1,
            targets2
        )
        teacher_logits_shuffle = pseudo_labels
        

    return inputs_shuffle, teacher_logits_shuffle



def train_distill(epoch, train_loader, module_list, optimizer, opt, problematic_classes = None):
    """One epoch distillation using practise.py approach"""
    # set modules as train()
    for module in module_list:
        module.cuda()
        module.eval()
        module.get_feat = 'pre_GAP'
    # set teacher as eval()
    module_list[-1].eval()

    # Use MSE loss like practise.py
    alpha = 0.5
    gamma = 0.5
    criterion_kd = MIXSTDLoss(opt, alpha, gamma)
    # criterion = torch.nn.MSELoss(reduction='mean')

    model_s = module_list[0]  # Student model
    model_t = module_list[-1]  # Teacher model
    finish = False
    iter_nums = 0
    synthetic_inputs_list = []
    synthetic_targets_list = []

    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        from torch.utils.data import TensorDataset
        for idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > epoch:
                finish = True
                break
            # print(data)
            # print(len(data))
            # print(data[0].shape)
            input = data.cuda()
            target = target.cuda()
                
            data_time.update(time.time() - end)
            
            # Determine number of classes based on dataset
            if hasattr(opt, 'dataset') and opt.dataset == 'cifar100':
                num_classes = 100
            else:
                num_classes = 1000  # Default for ImageNet
                
            if opt.pmixup:
                inputs_shuffle, targets_shuffle = PMU(input, target, opt.partmixup, opt.beta_a, num_classes=num_classes)
            else:            
                inputs_shuffle, targets_shuffle = FMU(input, target, opt.beta_a, module_list[-1], num_classes=num_classes, problematic_classes=problematic_classes)
                
            inputs_shuffle = inputs_shuffle.float()
            if torch.cuda.is_available():
                inputs_shuffle = inputs_shuffle.cuda()
                targets_shuffle = targets_shuffle.cuda()
            combined_inputs = torch.cat([input, inputs_shuffle], dim=0)  # Shape: [2*B, C, H, W]
            
            synthetic_inputs_list.append(inputs_shuffle.detach().cpu())
            synthetic_targets_list.append(targets_shuffle.detach().cpu())
            
            # ===================forward=====================
            # Get features and logits from both models (like practise.py)
            with torch.no_grad():
                t_logits, t_features = model_t(inputs_shuffle)
                # print(f"train_distill - Teacher model output type: {type(t_logits)}")
                # print(f"train_distill - Teacher model output shape: {t_logits.shape if hasattr(t_logits, 'shape') else 'No shape'}")
                # if isinstance(t_logits, tuple):
                #     print(f"train_distill - Teacher model output tuple length: {len(t_logits)}")
                #     for i, item in enumerate(t_logits):
                #         print(f"train_distill - Tuple item {i} shape: {item.shape}")
                t_probs = nn.functional.softmax(t_logits / 1.0, dim=1)

            optimizer.zero_grad()
            s_logits, s_features = model_s(inputs_shuffle)
            s_log_probs = nn.functional.log_softmax(s_logits / 1.0, dim=1)

            # Use MSE loss between features (like practise.py)
            # loss = criterion(s_features, t_features)
            loss = criterion_kd(s_logits, t_logits, target)
            

            # Calculate accuracy
            s_logits=s_logits.cuda()
            acc1, acc5 = accuracy(s_logits, target, topk=(1, 5))
            losses.update(loss.item(), inputs_shuffle.size(0))
            top1.update(acc1[0], inputs_shuffle.size(0))
            top5.update(acc5[0], inputs_shuffle.size(0))

            # ===================backward=====================
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if iter_nums % 50 == 0:
                print('Epoch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    iter_nums, epoch, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    all_inputs = torch.cat(synthetic_inputs_list, dim=0)
    all_targets = torch.cat(synthetic_targets_list, dim=0)
    synthetic_dataset = TensorDataset(all_inputs, all_targets)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=train_loader.batch_size, shuffle=True)

    return top1.avg, losses.avg, synthetic_loader


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #vis_feator = FeatureVisualizer()
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            # output std scaling 
            std = torch.std(output, dim=-1, keepdim=True)
            output = output/std
        
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

