import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ce_loss(logits, targets, use_hard_labels=True, weight=None, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, weight=weight, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        if weight == None:
           nll_loss = torch.sum(-targets * log_pred, dim=1)
        else:
           nll_loss = torch.sum(-targets * log_pred * weight, dim=1)
        return nll_loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target, reduction):
        index = torch.zeros_like(x, dtype=torch.uint8).to(self.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        if reduction != None:
           return F.cross_entropy(self.s*output, target, weight=self.weight)
        else:
           return F.cross_entropy(self.s*output, target, weight=self.weight, reduction=reduction)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)    ## make adjacency matrix with diagnoals = 1

        contrast_count = features.shape[1]  ## 1 ??
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  ## removed the unused dimension

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # compute correlations
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   ## final adjacency matrix

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos_numer = (mask * log_prob).sum(1)
        mean_log_prob_pos_denom = mask.sum(1)
        temp = mask.sum(1)
        nonzero_num = temp.shape[0] - torch.sum(temp == 0)
        mean_log_prob_pos_numer = mean_log_prob_pos_numer[temp != 0]
        mean_log_prob_pos_denom = mean_log_prob_pos_denom[temp != 0]
        mean_log_prob_pos = mean_log_prob_pos_numer / mean_log_prob_pos_denom

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.view(anchor_count, nonzero_num).mean()
        #loss = torch.mean(loss)
        return loss
