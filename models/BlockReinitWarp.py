import torch
import torch.nn as nn
from typing import Dict, Set, Union, List

class BlockReinitWarp(nn.Module):
    def __init__(self, model) -> None:
        super(BlockReinitWarp, self).__init__()
        self.model = model
        self.get_feat = "None"
        self.named_modules_dict = dict(model.named_modules())
        self.trainable_blocks: Set[str] = set()
        self.reinitialized_blocks: Set[str] = set()
        
        print("\nAvailable modules in pruned model:")
        for name in self.named_modules_dict.keys():
            if isinstance(self.named_modules_dict[name], nn.Module):
                print(f"  {name}")

    def _get_blocks_in_layer(self, layer_num: int) -> List[str]:
        """Get all blocks in a specific layer of the pruned model."""
        blocks = []
        prefix = f'model.layer{layer_num}.'
        for name in self.named_modules_dict.keys():
            if len(name.split('.')) == 3:  # Only get the block level  name.startswith(prefix) and 
                blocks.append(name)
        return sorted(blocks, key=lambda x: int(x.split('.')[-1]))
    

    def _get_original_block_number(self, layer_num: int, pruned_block_num: int, rm_blocks: List[str]) -> int:
        """Convert a pruned model's block number to its original number before pruning."""
        removed_before = sum(1 for block in rm_blocks 
                           if block.startswith(f'layer{layer_num}.') 
                           and int(block.split('.')[1]) <= pruned_block_num)
        return pruned_block_num + removed_before

    def _get_pruned_block_number(self, layer_num: int, original_block_num: int, rm_blocks: List[str]) -> int:
        """Convert an original block number to its number in the pruned model."""
        removed_before = sum(1 for block in rm_blocks 
                           if block.startswith(f'layer{layer_num}.') 
                           and int(block.split('.')[1]) < original_block_num)
        return original_block_num - removed_before

    def _get_adjacent_blocks(self, rm_block: str, rm_blocks: List[str], distance: int = 1) -> List[str]:
        """Get adjacent blocks: nearest non-pruned block in the same layer."""
        # Parse the removed block's layer and block numbers
        layer_num = int(rm_block[5])  # e.g., 'layer1.1' -> 1
        block_num = int(rm_block[7])  # e.g., 'layer1.1' -> 1
        
        adjacent_blocks = []
        
        # Get the previous block in the same layer if it exists
        # prev_block = f'model.layer{layer_num}.{block_num-1}'
        # if prev_block in self.named_modules_dict and block_num > 0:
        #     if prev_block not in [f'model.{b}' if not b.startswith('model.') else b for b in rm_blocks]:
        #         adjacent_blocks.append(prev_block)
        #         print(f"Found previous block in same layer: {prev_block}")

        current_block = block_num - 1
        while current_block >= 0:
            prev_block = f'model.layer{layer_num}.{current_block}'
            current_block_name = f'layer{layer_num}.{current_block}'
            
            # Check if this block exists and is not in rm_blocks
            if prev_block in self.named_modules_dict:
                if current_block_name not in rm_blocks and prev_block not in [f'model.{b}' if not b.startswith('model.') else b for b in rm_blocks]:
                    adjacent_blocks.append(prev_block)
                    print(f"Found previous block in same layer: {prev_block}")
                    break
            current_block -= 1
        # prev_block = f'model.layer{layer_num}.{block_num-2}'
        # if prev_block in self.named_modules_dict and block_num > 0:
        #     if prev_block not in [f'model.{b}' if not b.startswith('model.') else b for b in rm_blocks]:
        #         adjacent_blocks.append(prev_block)
        #         print(f"Found previous block in same layer: {prev_block}")
        
        # # Get the first block of the next layer if it exists
        # next_layer_block = f'model.layer{layer_num+1}.0'
        # if next_layer_block in self.named_modules_dict:
        #     if next_layer_block not in [f'model.{b}' if not b.startswith('model.') else b for b in rm_blocks]:
        #         adjacent_blocks.append(next_layer_block)
        #         print(f"Found first block of next layer: {next_layer_block}")
        # next_layer_block = f'model.layer{layer_num+1}.1'
        # if next_layer_block in self.named_modules_dict:
        #     if next_layer_block not in [f'model.{b}' if not b.startswith('model.') else b for b in rm_blocks]:
        #         adjacent_blocks.append(next_layer_block)
        #         print(f"Found first block of next layer: {next_layer_block}")
        
        return adjacent_blocks

    def reinitialize_nearby_blocks(self, rm_blocks: Union[str, List[str]], distance: int = 1):
        """Reinitialize blocks within the specified distance of the removed blocks."""
        if isinstance(rm_blocks, str):
            rm_blocks = [rm_blocks]
        
        print(f"\nProcessing removed blocks: {rm_blocks}")
        
        for rm_block in rm_blocks:
            print(f"\nProcessing block: {rm_block}")
            
            # Get adjacent blocks to reinitialize
            adjacent_blocks = self._get_adjacent_blocks(rm_block, rm_blocks, distance)
            print(f"Adjacent blocks to reinitialize: {adjacent_blocks}")
            
            # Reinitialize each adjacent block
            for block_name in adjacent_blocks:
                if block_name in self.named_modules_dict:
                    block = self.named_modules_dict[block_name]
                    print(f"Reinitializing block: {block_name}")
                    
                    # # Reinitialize the block's parameters
                    # for name, m in block.named_modules():
                    #     if isinstance(m, nn.Conv2d):
                    #         print(f"  Reinitializing Conv2d: {name}")
                    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    #         if m.bias is not None:
                    #             nn.init.constant_(m.bias, 0)
                    #     elif isinstance(m, nn.BatchNorm2d):
                    #         print(f"  Reinitializing BatchNorm2d: {name}")
                    #         # nn.init.constant_(m.weight, 1)
                    #         # nn.init.constant_(m.bias, 0)
                    
                    # Store without 'model.' prefix for parameter tracking
                    clean_name = block_name.replace('model.', '')
                    self.reinitialized_blocks.add(clean_name)
                    self.trainable_blocks.add(clean_name)
                    print(f"Added {clean_name} to trainable blocks")

        print(f"\nFinal reinitialized blocks: {self.reinitialized_blocks}")
        print(f"Final trainable blocks: {self.trainable_blocks}")

    def freeze_non_trainable_blocks(self):
        """Freeze all parameters except those in trainable blocks."""
        print("\nFreezing/Unfreezing parameters:")
        trainable_count = 0
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Get the block name from the parameter name
            parts = name.split('.')
            if len(parts) >= 3:
                # Handle both regular block parameters and downsample parameters
                if 'downsample' in parts:
                    # For downsample layers, always keep them trainable if they're in a trainable block's parent
                    block_name = '.'.join(parts[1:3])  # e.g., 'layer1.0' for 'model.layer1.0.downsample.0.weight'
                    should_train = block_name in self.trainable_blocks or any(
                        b.startswith(f"{parts[1]}.{parts[2]}") for b in self.trainable_blocks
                    )
                else:
                    block_name = '.'.join(parts[1:3])  # e.g., 'layer1.0' for 'model.layer1.0.conv1.weight'
                    should_train = block_name in self.trainable_blocks
            else:
                # Handle parameters not in blocks (like initial conv layer)
                block_name = parts[1] if len(parts) > 1 else parts[0]
                should_train = block_name in self.trainable_blocks
            
            param.requires_grad = should_train
            if should_train:
                trainable_count += 1
                print(f"  UNFROZEN: {name} (block: {block_name})")
            else:
                frozen_count += 1
                print(f"  Frozen: {name} (block: {block_name})")
        
        print(f"\nTotal parameters: {trainable_count + frozen_count}")
        print(f"Trainable parameters: {trainable_count}")
        print(f"Frozen parameters: {frozen_count}")

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        params = []
        print("\nCollecting trainable parameters:")
        
        for name, param in self.model.named_parameters():
            # Get the block name from the parameter name
            parts = name.split('.')
            if len(parts) >= 3:
                # Handle both regular block parameters and downsample parameters
                if 'downsample' in parts:
                    # For downsample layers, associate them with their parent block
                    block_name = '.'.join(parts[1:3])  # e.g., 'layer1.0' for 'model.layer1.0.downsample.0.weight'
                else:
                    block_name = '.'.join(parts[1:3])  # e.g., 'layer1.0' for 'model.layer1.0.conv1.weight'
            else:
                # Handle parameters not in blocks (like initial conv layer)
                block_name = parts[1] if len(parts) > 1 else parts[0]
                
            if block_name in self.trainable_blocks:
                params.append(param)
                print(f"  Added parameters from: {name} (block: {block_name})")
        
        if not params:
            print("WARNING: No trainable parameters found!")
        else:
            print(f"Found {len(params)} trainable parameter tensors")
            
        return params

    def forward(self, x):
        self.model.get_feat = self.get_feat
        return self.model(x) 