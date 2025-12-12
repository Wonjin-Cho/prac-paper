from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIXSTDLoss(nn.Module):
    def __init__(self,opt, alpha, gamma):
        super(MIXSTDLoss, self).__init__()
        self.opt = opt
        self.cross_ent = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.KL = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, logit_s, logit_t, target):

        stdt = torch.std(logit_t, dim=-1,keepdim=True)
        stds = torch.std(logit_s, dim=-1, keepdim=True)
        target_one_hot = F.one_hot(target, num_classes=logit_s.shape[1]).float()

        ## CLS ##        
        loss = -F.log_softmax(logit_s/stds,-1) * target_one_hot
        loss_cls = self.gamma * (torch.sum(loss))/logit_s.shape[0]        
        ## STD KD ## 
        p_s = F.log_softmax(logit_s/stds, dim=1)
        p_t = F.softmax(logit_t/stdt, dim=1)
        std_KD = self.KL(p_s, p_t) 
        loss_div = self.alpha * std_KD
        tot_loss = loss_cls+loss_div
        return tot_loss
