import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock, Bottleneck
import copy
import collections
import types
from models.AdaptorWarp import AdaptorWarp
import numpy as np
torch.manual_seed(487)  # Set PyTorch seed

def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def compute_importance_resnet(model, method='l2', use_cuda=True):
    """
    ResNet의 Conv 레이어에서 필터 중요도를 계산합니다.

    Args:
        model: ResNet 모델 객체.
        method: 중요도 계산 방법 ('l2', 'l1').
        use_cuda: GPU 사용 여부.
    
    Returns:
        각 Conv 레이어의 중요도를 포함하는 딕셔너리 {레이어 이름: 중요도 리스트}.
    """
    importance_dict = {}

    # 모델의 모든 레이어를 순회
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            #print(f"Layer: {name}, Weights shape: {weights.shape}")  # 디버깅용
            
            # 중요도 계산: axis=(1, 2, 3)을 사용 (필터별 중요도)
            if method == 'l2':
                importance = np.linalg.norm(weights, axis=(1, 2, 3))  # 필터별 L2 norm 계산
            elif method == 'l1':
                importance = np.sum(np.abs(weights), axis=(1, 2, 3))  # 필터별 L1 norm 계산
            else:
                raise ValueError(f"Unknown importance calculation method: {method}")
            importance = sum(importance)/len(importance)
            importance = float(importance)
            # 중요도를 딕셔너리에 저장
            importance_dict[name] = importance

    return importance_dict

def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1):
    eps = 1e-10
    keep_ratio = 1 - ratio
    old_net = net

    #net = copy.deepcopy(net)
    net.zero_grad()
    weights = []
    mean = 0
    fc_layers = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear):
                fc_layers.append(layer)
            weights.append(layer.weight)
            
    #nn.init.xavier_normal(fc_layers[-1].weight)
    
    inputs_one = []
    targets_one = []

    grad_w = None
    grad_f = None
    for w in weights:
        w.requires_grad_(True)

    dataloader_iter = iter(train_dataloader)
    for it in range(num_iters):
        # print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = next(dataloader_iter)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)

        start = 0
        intv = 20

        while start < N:
            end = min(start + intv, N)
            # print('(1):  %d -> %d.' % (start, end))
            inputs_one.append(din[start:end])
            targets_one.append(dtarget[start:end])
            
            outputs = net.forward(inputs[start:end].to(device))# / 200  # divide by temperature
            if isinstance(outputs,tuple):
                outputs=outputs[0]
            outputs = outputs/200
            loss = F.cross_entropy(outputs, targets[start:end].to(device))
            grad_w_p = autograd.grad(loss, weights, create_graph=False, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            start = end

    for it in range(len(inputs_one)):
        # print("(2): Iterations %d/%d." % (it, len(inputs_one)))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        outputs = net.forward(inputs)# / 200  # divide by temperature
        if isinstance(outputs,tuple):
            outputs=outputs[0]
        outputs = outputs/200

        loss = F.cross_entropy(outputs, targets)
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1
        z.backward()

    # Calculate acceptable scores per layer
    layer_scores = {}  # Store the sum of acceptable scores for each layer
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            score = (-layer.weight.data * layer.weight.grad).sum().item()  # Sum the scores for the layer
            score += layer.weight.mean().item()
            layer_scores[old_modules[idx]] = score  # Map old net layers to their scores

    # Calculate the sum of acceptable scores for layers
    total_score_sum = sum(layer_scores.values())
    print(total_score_sum)
    print(f"total score sum of Grasp = {total_score_sum:0.4f}")

    return layer_scores

def score_blocks_by_activation_change(model, data_loader, num_batches=10, device='cuda'):
    """
    Measure a score per block based on the change in number of active activations
    (values > 0 after ReLU) between each block and its previous block.

    Args:
        model: PyTorch ResNet model
        data_loader: DataLoader to feed inputs
        num_batches: Number of batches to process for estimation
        device: 'cuda' or 'cpu'

    Returns:
        dict: {block_name -> activation_change_from_previous}
              The first block's change is 0.0 since it has no previous block.
        dict: {block_name -> total_active_count}
    """
    model.eval()
    if device:
        model.to(device)

    # Identify top-level residual blocks in order
    block_names_in_order = []
    name_to_module = {}
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            # Keep only the top-level block name like 'layer1.0'
            block_name = '.'.join(name.split('.')[:2])
            if block_name not in name_to_module:
                name_to_module[block_name] = module
                block_names_in_order.append(block_name)

    # Deduplicate while preserving order in case of nested reports
    seen = set()
    ordered_blocks = []
    for bn in block_names_in_order:
        if bn not in seen:
            seen.add(bn)
            ordered_blocks.append(bn)

    # Register hooks on the block modules to capture post-block outputs
    activation_sums = collections.OrderedDict((bn, 0.0) for bn in ordered_blocks)
    hooks = []

    def make_hook(block_name):
        def hook(_m, _inp, out):
            if not torch.is_tensor(out):
                return
            # Count number of positive activations in the block output
            active = (out > 0).sum().item()
            activation_sums[block_name] += active
        return hook

    # Attach hooks to the actual block module objects
    for block_name in ordered_blocks:
        module = dict(model.named_modules())[block_name]
        hooks.append(module.register_forward_hook(make_hook(block_name)))

    # Run a few batches to accumulate counts
    with torch.no_grad():
        for batch_index, (images, _targets) in enumerate(data_loader):
            if batch_index >= num_batches:
                break
            if device:
                images = images.to(device)
            _ = model(images)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Compute differences relative to previous block
    activation_change = {}
    for idx, block_name in enumerate(ordered_blocks):
        if idx == 0:
            activation_change[block_name] = 0.0
        else:
            prev_name = ordered_blocks[idx - 1]
            activation_change[block_name] = activation_sums[block_name] - activation_sums[prev_name]

    return dict(activation_sums) #activation_change#, dict(activation_sums)

def metric(metric_loader, model, origin_model, trained = False):
    criterion = torch.nn.MSELoss(reduction='mean')

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    origin_model.get_feat = 'pre_GAP'
    model.cuda()
    model.eval()
    model.get_feat = 'pre_GAP'
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    origin_accuracies = AverageMeter()

    # Initialize per-category accuracy tracking
    num_classes = 1000  # Assuming ImageNet classes
    correct_per_class = torch.zeros(num_classes).cuda()
    total_per_class = torch.zeros(num_classes).cuda()
    origin_correct_per_class = torch.zeros(num_classes).cuda()
    # Initialize per-class MSE loss tracking
    mse_loss_per_class = torch.zeros(num_classes).cuda()

    end = time.time()
    for i, (data, target) in enumerate(metric_loader):
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)
            loss = criterion(s_features, t_features)
            
            # Calculate overall accuracy
            acc = accuracy(s_output, target, topk=(1,))[0]
            origin_acc = accuracy(t_output, target, topk=(1,))[0]
            
            # # Calculate per-class accuracy and accumulate per-class MSE
            # _, predicted = s_output.max(1)
            # _, origin_predicted = t_output.max(1)
            
            # # Update per-class statistics
            # for j in range(num_classes):
            #     mask = (target == j)
            #     if mask.sum() > 0:
            #         correct_per_class[j] += (predicted[mask] == j).sum()
            #         total_per_class[j] += mask.sum()
            #         origin_correct_per_class[j] += (origin_predicted[mask] == j).sum()
            #         # Per-class MSE: accumulate sum of MSE for all samples of class j
            #         class_mse = criterion(s_features[mask], t_features[mask])
            #         mse_loss_per_class[j] += class_mse.item() * mask.sum().item()

        losses.update(loss.data.item(), data.size(0))
        accuracies.update(acc.item(), data.size(0))
        origin_accuracies.update(origin_acc.item(), data.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Metric: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Accuracy {accuracies.val:.2f} ({accuracies.avg:.2f})'.format(
                   i, len(metric_loader), batch_time=batch_time,
                   data_time=data_time, losses=losses, accuracies=accuracies))

    print(' * Metric Loss {loss.avg:.4f}'.format(loss=losses))

    problematic_classes = []
    # # Calculate per-class accuracy and differences
    # class_accuracies = []
    # avg_mse_per_class = []
    # for j in range(num_classes):
    #     if total_per_class[j] > 0:
    #         pruned_acc = (correct_per_class[j] / total_per_class[j] * 100).item()
    #         origin_acc = (origin_correct_per_class[j] / total_per_class[j] * 100).item()
    #         diff = origin_acc - pruned_acc  # Positive means pruned model is worse
    #         class_accuracies.append((j, pruned_acc, origin_acc, diff))
    #         # Compute average MSE for this class
    #         avg_mse = mse_loss_per_class[j].item() / total_per_class[j].item()
    #         avg_mse_per_class.append((j, avg_mse))
    
    # # Sort by difference (largest to smallest)
    # class_accuracies.sort(key=lambda x: (x[3]), reverse=True)
    
    # # Get classes with large accuracy differences (>30%)
    # problematic_classes = [j for j, _, _, diff in class_accuracies if diff > 10]
    
    # if trained:
    #     # Create timestamp for unique filename
    #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     log_filename = f'class_accuracy_comparison_{timestamp}.log'
    #     mse_log_filename = f'class_mse_comparison_{timestamp}.log'
        
    #     # Save results to log file (accuracy)
    #     with open(log_filename, 'w') as f:
    #         f.write("Per-Class Accuracy Comparison\n")
    #         f.write("=" * 80 + "\n")
    #         f.write("Class\tPruned Model\tOriginal Model\tDifference\n")
    #         f.write("-" * 80 + "\n")
            
    #         for class_idx, pruned_acc, origin_acc, diff in class_accuracies:
    #             f.write(f"{class_idx}\t{pruned_acc:.2f}%\t\t{origin_acc:.2f}%\t\t{diff:+.2f}%\n")
            
    #         f.write("=" * 80 + "\n")
    #         f.write(f"Overall Accuracy - Pruned: {accuracies.avg:.2f}%, Original: {origin_accuracies.avg:.2f}%\n")
    #         f.write(f"Average Absolute Difference: {sum((x[3]) for x in class_accuracies)/len(class_accuracies):.2f}%\n")
    #         f.write(f"\nProblematic Classes (diff > 30%): {problematic_classes}\n")
        
    #     # Save per-class MSE results to log file
    #     with open(mse_log_filename, 'w') as f:
    #         f.write("Per-Class MSE Loss Comparison\n")
    #         f.write("=" * 80 + "\n")
    #         f.write("Class\tAvg MSE Loss\n")
    #         f.write("-" * 80 + "\n")
    #         for class_idx, avg_mse in avg_mse_per_class:
    #             f.write(f"{class_idx}\t{avg_mse:.6f}\n")
    #         f.write("=" * 80 + "\n")
    #         f.write(f"Average MSE Loss (all classes): {sum(x[1] for x in avg_mse_per_class)/len(avg_mse_per_class):.6f}\n")
        
    #     print(f"\nPer-class accuracy comparison has been saved to: {log_filename}")
    #     print(f"Per-class MSE loss comparison has been saved to: {mse_log_filename}")
    #     print(f"Problematic Classes (diff > 10%): {problematic_classes}")
    
    print(f"Overall Accuracy - Pruned: {accuracies.avg:.2f}%, Original: {origin_accuracies.avg:.2f}%")
    
    return losses.avg, accuracies.avg, origin_accuracies.avg, problematic_classes

def measure_feature_reconstruction_error(model, data_loader, num_batches=10, device='cuda'):
    """
    Compute per-block importance by replacing a block's output with its input
    (i.e., pass previous block's output forward, effectively bypassing the block)
    and measuring the L2 change in final logits.

    I(B) = || Z_original - Z_bypassed ||_2, averaged per sample.
    Higher I(B) => more important block. Lower I(B) => more redundant/unimportant.

    Args:
        model: PyTorch model (e.g., ResNet)
        data_loader: DataLoader providing input batches
        num_batches: Number of batches to evaluate
        device: Device string ('cuda' or 'cpu')

    Returns:
        OrderedDict: {block_name: average_l2_change}
    """
    model.eval()
    if device:
        model.to(device)

    # Identify ordered top-level residual blocks
    block_names_in_order = []
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            block_name = '.'.join(name.split('.')[:2])
            if block_name not in block_names_in_order:
                block_names_in_order.append(block_name)

    # Deduplicate preserving order
    seen = set()
    ordered_blocks = []
    for bn in block_names_in_order:
        if bn not in seen:
            seen.add(bn)
            ordered_blocks.append(bn)

    # Helper to get logits (handle models returning (logits, features))
    def get_logits(x):
        out = model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    l2_change_sum = collections.OrderedDict((bn, 0.0) for bn in ordered_blocks)
    sample_count = 0

    with torch.no_grad():
        for batch_index, (images, _targets) in enumerate(data_loader):
            if batch_index >= num_batches:
                break
            if device:
                images = images.to(device)

            # First pass: baseline logits and capture each block's input
            saved_inputs = {}
            hooks_cap = []
            def make_input_capture_hook(block_name):
                def _cap(m, inp, out):
                    # Save the block input (previous block's output)
                    if isinstance(inp, (tuple, list)) and len(inp) > 0 and torch.is_tensor(inp[0]):
                        saved_inputs[block_name] = inp[0]
                return _cap
            for block_name in ordered_blocks:
                module = dict(model.named_modules())[block_name]
                hooks_cap.append(module.register_forward_hook(make_input_capture_hook(block_name)))
            z_orig = get_logits(images)
            for h in hooks_cap:
                h.remove()

            N = z_orig.size(0)
            sample_count += N

            # Second pass per block: bypass the block by returning its saved input
            for block_name in ordered_blocks:
                # Skip if we could not capture input for this block (e.g., first block before any activation)
                if block_name not in saved_inputs:
                    continue
                module = dict(model.named_modules())[block_name]

                saved_inp = saved_inputs[block_name]
                def _bypass(_m, _inp, out):
                    if torch.is_tensor(saved_inp):
                        # If shape mismatches, let it raise so we can skip this block gracefully
                        if out is not None and torch.is_tensor(out) and out.shape != saved_inp.shape:
                            return out  # keep original to avoid shape crash
                        return saved_inp
                    return out

                h = module.register_forward_hook(_bypass)
                try:
                    z_byp = get_logits(images)
                    diff = (z_orig - z_byp).view(N, -1)
                    l2 = diff.norm(p=2, dim=1).sum().item()
                    l2_change_sum[block_name] += l2
                except Exception:
                    # If bypassing this block breaks shapes/graph, skip contribution for this batch
                    pass
                finally:
                    h.remove()

    if sample_count > 0:
        for k in l2_change_sum:
            l2_change_sum[k] = l2_change_sum[k] / sample_count

    return l2_change_sum


def measure_fisher_information_per_block(model, data_loader, num_batches=10, device='cuda'):
    """
    Estimate Fisher Information per residual block by accumulating squared
    gradients of the log-likelihood with respect to parameters in each block.

    For classification and a batch (images, targets):
      - Compute log-prob of the true classes
      - Backpropagate the SUM of true-class log-probs to get gradients
      - Square gradients, sum over parameters belonging to each block
      - Accumulate over batches and average per sample processed

    Returns an OrderedDict mapping block name (e.g., 'layer1.0') to its Fisher score.
    Higher Fisher => more important block.
    """
    model.eval()  # keep BN/Dropout deterministic; gradients still computed
    if device:
        model.to(device)

    # Collect residual block names in order
    block_names = []
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            blk = '.'.join(name.split('.')[:2])
            if blk not in block_names:
                block_names.append(blk)

    fisher_sums = collections.OrderedDict((bn, 0.0) for bn in block_names)
    sample_count = 0

    batch_idx = 0
    for images, targets in data_loader:
        if batch_idx >= num_batches:
            break
        batch_idx += 1

        if device:
            images = images.to(device)
            targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        log_probs = F.log_softmax(logits, dim=1)
        true_log_prob_sum = log_probs.gather(1, targets.view(-1, 1)).squeeze(1).sum()

        # Backpropagate sum of log-likelihoods (per-sample) to accumulate Fisher properly
        true_log_prob_sum.backward()

        # Aggregate squared gradients per block
        for pname, p in model.named_parameters():
            if p.grad is None:
                continue
            parts = pname.split('.')
            blk = None
            if len(parts) >= 2 and parts[0].startswith('layer'):
                blk = parts[0] + '.' + parts[1]
            if blk is None or blk not in fisher_sums:
                continue
            fisher_sums[blk] += (p.grad.detach() ** 2).sum().item()

        sample_count += images.size(0)

    # Average per sample processed
    if sample_count > 0:
        for k in fisher_sums:
            fisher_sums[k] = fisher_sums[k] / sample_count

    return fisher_sums


###def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1):
    eps = 1e-10
    keep_ratio = 1 - ratio
    old_net = net

    net = copy.deepcopy(net)
    net.zero_grad()
    weights = []
    fc_layers = []

    # Identify main layers, preconv, and afterconv layers
    if isinstance(net, AdaptorWarp):
        adaptor = net
        net = net.model  # Extract the actual model
    else:
        adaptor = None

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear):
                fc_layers.append(layer)
            weights.append(layer.weight)

    # Include preconv and afterconv layers if available
    if adaptor is not None:
        for preconv in adaptor.name2preconvs.values():
            weights.append(preconv.weight)
        for afterconv in adaptor.name2afterconvs.values():
            weights.append(afterconv.weight)

    if len(fc_layers) > 0:
        nn.init.xavier_normal_(fc_layers[-1].weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    grad_f = None
    for w in weights:
        w.requires_grad_(True)

    dataloader_iter = iter(train_dataloader)
    for it in range(num_iters):
        inputs, targets = next(dataloader_iter)
        N = inputs.shape[0]
        start = 0
        intv = 20

        while start < N:
            end = min(start + intv, N)
            inputs_one.append(inputs[start:end])
            targets_one.append(targets[start:end])

            outputs = net.forward(inputs[start:end].to(device))
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Ensure outputs are tensor

            loss = F.cross_entropy(outputs, targets[start:end].to(device))
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            start = end

    for it in range(len(inputs_one)):
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        outputs = net.forward(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Ensure outputs are tensor
        outputs = outputs / 200

        loss = F.cross_entropy(outputs, targets)
        grad_f = autograd.grad(loss, weights, create_graph=True)

        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1

        # Include preconv and afterconv gradients
        if adaptor is not None:
            for preconv in adaptor.name2preconvs.values():
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1
            for afterconv in adaptor.name2afterconvs.values():
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1

        z.backward()

    grads = {}
    block_mapping = {}  # Maps layers to their parent blocks
    block_scores = {}  # Stores scores per block

    # Identify blocks and assign layers to blocks
    for module in net.modules():
        if isinstance(module, (BasicBlock, Bottleneck)):  # Identify ResNet blocks
            grads[module] = 0  # Initialize score for this block
            for sub_layer in module.children():  # Assign sub-layers to their block
                if isinstance(sub_layer, (nn.Conv2d, nn.Linear)):
                    block_mapping[sub_layer] = module

    # Include adaptors in block mapping
    if adaptor is not None:
        for name, preconv in adaptor.name2preconvs.items():
            block_mapping[preconv] = None  # Treat as independent if no clear parent
        for name, afterconv in adaptor.name2afterconvs.items():
            block_mapping[afterconv] = None

    # Compute block-wise scores
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            block = block_mapping.get(layer, None)
            if block:
                grads[block] += (-layer.weight.data * layer.weight.grad).sum()  # Accumulate per block

    # Include preconv and afterconv in score computation
    if adaptor is not None:
        for preconv in adaptor.name2preconvs.values():
            grads[preconv] = (-preconv.weight.data * preconv.weight.grad).sum()
        for afterconv in adaptor.name2afterconvs.values():
            grads[afterconv] = (-afterconv.weight.data * afterconv.weight.grad).sum()

    # Normalize block-wise scores
    block_scores = {}
    for block, score in grads.items():
        block_scores[block] = score

    return block_scores
###

# def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1):
#     eps = 1e-10
#     keep_ratio = 1 - ratio
#     old_net = net

#     net = copy.deepcopy(net)
#     net.zero_grad()
#     weights = []
    
#     fc_layers = []
#     for layer in net.modules():
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             if isinstance(layer, nn.Linear):
#                 fc_layers.append(layer)
#             weights.append(layer.weight)
#     # if len(fc_layers) > 0:
#     #     nn.init.xavier_normal_(fc_layers[-1].weight)

#     inputs_one = []
#     targets_one = []

#     grad_w = None
#     grad_f = None
#     for w in weights:
#         w.requires_grad_(True)

#     dataloader_iter = iter(train_dataloader)
#     for it in range(num_iters):
#         inputs, targets = next(dataloader_iter)
#         N = inputs.shape[0]
#         start = 0
#         intv = 20

#         while start < N:
#             end = min(start + intv, N)
#             inputs_one.append(inputs[start:end])
#             targets_one.append(targets[start:end])

#             outputs = net.forward(inputs[start:end].to(device))
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]  # Ensure outputs are tensor

#             loss = F.cross_entropy(outputs, targets[start:end].to(device))
#             grad_w_p = autograd.grad(loss, weights, create_graph=False)
#             if grad_w is None:
#                 grad_w = list(grad_w_p)
#             else:
#                 for idx in range(len(grad_w)):
#                     grad_w[idx] += grad_w_p[idx]
#             start = end

#     for it in range(len(inputs_one)):
#         inputs = inputs_one.pop(0).to(device)
#         targets = targets_one.pop(0).to(device)
#         outputs = net.forward(inputs)
#         if isinstance(outputs, tuple):
#             outputs = outputs[0]  # Ensure outputs are tensor
#         outputs = outputs / 200

#         loss = F.cross_entropy(outputs, targets)
#         grad_f = autograd.grad(loss, weights, create_graph=True)

#         z = 0
#         count = 0
#         for layer in net.modules():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 z += (grad_w[count] * grad_f[count]).sum()
#                 count += 1
#         z.backward()

#     grads = {}
#     block_mapping = {}  # Maps layers to their parent blocks
#     block_scores = {}  # Stores scores per block

#     # Identify blocks and assign layers to blocks
#     for module in net.modules():
#         if isinstance(module, (BasicBlock, Bottleneck)):  # Identify ResNet blocks
#             grads[module] = 0  # Initialize score for this block
#             for sub_layer in module.children():  # Assign sub-layers to their block
#                 if isinstance(sub_layer, (nn.Conv2d, nn.Linear)):
#                     block_mapping[sub_layer] = module

#     # Compute block-wise scores
#     for idx, layer in enumerate(net.modules()):
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             block = block_mapping.get(layer, None)
#             if block:
#                 grads[block] += (-layer.weight.data * layer.weight.grad).sum()  # Accumulate per block

#     # Normalize block-wise scores
#     block_scores = {}
#     for block, score in grads.items():
#         block_scores[block] = score

#     return block_scores



### def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)
    net.zero_grad()
    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    fc_layers = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear):
                fc_layers.append(layer)
            weights.append(layer.weight)
    nn.init.xavier_normal(fc_layers[-1].weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    grad_f = None
    for w in weights:
        w.requires_grad_(True)

    intvs = {
        'cifar10': 128,
        'cifar100': 256,
        'tiny_imagenet': 128,
        'imagenet': 20
    }
    print_once = False
    dataloader_iter = iter(train_dataloader)
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = next(dataloader_iter)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)

        start = 0
        intv = 20

        while start < N:
            end = min(start+intv, N)
            print('(1):  %d -> %d.' % (start, end))
            inputs_one.append(din[start:end])
            targets_one.append(dtarget[start:end])
            #error happening here
            outputs = net.forward(inputs[start:end].to(device))  # divide by temperature to make it uniform
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract logits from tuple
            if print_once:
                x = F.softmax(outputs)
                print(x)
                print(x.max(), x.min())
                print_once = False
            loss = F.cross_entropy(outputs, targets[start:end].to(device))
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            start = end

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, len(inputs_one)))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        outputs = net.forward(inputs)# / 200  # divide by temperature to make it uniform
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Extract logits from tuple
        loss = F.cross_entropy(outputs, targets)
        
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:")
    print(norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    # print('** accept: ', acceptable_score)
    # keep_masks = dict()
    # for m, g in grads.items():
    #     keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return acceptable_score ###

