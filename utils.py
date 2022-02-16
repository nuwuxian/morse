import numpy as np
import torch
import datetime
import torch.nn.functional as F
import torch.nn as nn

def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)

def cal_simialrity(data, gt, num_class):
    # data: n * dim
    feat = []
    for i in range(num_class):
        idx = np.where(gt == i)[0]
        cluster_i = np.mean(data[idx], axis=0)
        # F1-normalize
        norm = np.linalg.norm(cluster_i)
        cluster_i /= norm
        feat.append(cluster_i)
    feat = np.array(feat)
    
    return np.matmul(feat, feat.T)


def refine_pesudo_label(xw, probs, threshold, prototype, model):
    # xw: batch_sz * dim
    # prototype: k * dim 

    # cal the cos-similiarity between xw and prototype
    # (batch_sz * dim) * (dim * k)
    feat_xw = model.forward_encoder(xw)
    feat_xw = F.normalize(feat_xw, dim=-1, p=2)

    simi = torch.matmul(feat_xw, prototype.T)
    proto_pred = torch.argmax(simi, -1)


    yu = torch.argmax(probs, -1)
    mask = (torch.max(probs, -1)[0] >= threshold).to(dtype=torch.float32)
    mask = torch.logical_and(mask, proto_pred == yu)

    return yu, mask


def update_proto(x, y, prototype, model):
    # prototype: k * dim
    # x: batch_sz * dim
    feat_x = model.forward_encoder(x)
    class_num = prototype.size(0)
    for feat, label in zip(feat_x, y):
        prototype[label] = prototype[label] * 0.999 + (1 - 0.999) * feat
    prototype = F.normalize(prototype, dim=-1, p=2)

    return prototype


def get_labeled_dist(dataset):
    counts = torch.unique(torch.tensor(dataset.targets), sorted=True, return_counts=True)[-1]
    return counts.float() / counts.sum()

def aug(x, x_bar, ratio):
    ret_x = x.copy()
    sz = x.shape[0]
    idx = np.random.choice(sz, int(sz * ratio))
    ret_x[idx] = x_bar[idx]
    return ret_x

# Debug info, clean ratio
def debug_label_info(pred, gt, num_class):
    cls_proportion = []
    for cls in range(num_class):
        idx = np.where(pred == cls)[0]
        if len(idx) == 0:
            print('Label %d is Empty' %cls)
            cls_proportion.append(0)
        else:
            cls_proportion.append(np.sum(gt[idx] == cls) * 1.0 / len(idx))

    return cls_proportion

def debug_real_label_info(pred, gt):
    idx = np.where(gt == 0)[0]
    cls_proportion = []

    for cls in range(12):
        proportion = np.sum(pred[idx] == cls) * 1.0 / len(idx)
        cls_proportion.append(proportion)
    return cls_proportion


def debug_unlabel_info(pred, gt, mask, num_class):
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    mask = mask.data.cpu().numpy()
    # select the true data
    idx = np.where(mask == True)[0]
    pred = pred[idx]
    gt = gt[idx]
    
    cls_clean = []
    cls_proportion = []

    for cls in range(num_class):
        cls_idx = np.where(pred == cls)[0]
        cls_proportion.append(len(cls_idx) * 1.0 / len(idx))

        if len(cls_idx) == 0:
            print('UnLabel %d is Empty' %cls)
            cls_clean.append(0)
        else:
            proportion = np.sum(gt[cls_idx] == cls) * 1.0 / len(cls_idx)
            cls_clean.append(proportion)

    return cls_clean + cls_proportion


def debug_real_unlabel_info(pred, gt, mask):
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    mask = mask.data.cpu().numpy()
    # select the true data
    idx = np.where(mask == True)[0]
    pred = pred[idx]
    gt = gt[idx]

    debug_ratio = []
    class_0_idx = np.where(gt == 0)[0]
    class_11_idx = np.where(gt == 11)[0]

    for cls in range(12):
        proportion = np.sum(pred[class_0_idx] == cls) * 1.0 / len(class_0_idx)
        debug_ratio.append(proportion)

    class_11_clean = np.sum(pred[class_11_idx] == 11) * 1.0 / len(class_11_idx)
    debug_ratio.append(class_11_clean)
    
    return debug_ratio

# Cosine learning rate scheduler.
#
# From https://github.com/valencebond/FixMatch_pytorch/blob/master/lr_scheduler.py
class WarmupCosineLrScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


# Recording Tools
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_prototype(labeled_loader, model, device, class_num):
    ret_feat, ret_label = [], []
    with torch.no_grad():
        for x, y in labeled_loader:
            x = x.to(device)
            y = y.to(device)
            feat = model.forward_encoder(x)
            ret_feat.append(feat)
            ret_label.append(y)
    ret_feat = torch.cat(ret_feat, dim=0).to(device)
    ret_label = torch.cat(ret_label, dim=0).to(device)
    ret_label = ret_label.data.cpu().numpy()

    prototype = []
    for cls in range(class_num):
        idx = np.where(ret_label == cls)[0]
        prototype.append(torch.mean(ret_feat[idx, :], dim=0))
    prototype = torch.vstack(prototype)
    prototype = F.normalize(prototype, dim=-1, p=2)

    return prototype

def predict_dataset_softmax(predict_loader, model, device, train_num, type):
    model.eval()
    if type == 'confidence':
        softmax_outs = []
        with torch.no_grad():
            for x, _, _ in predict_loader:
                x = x.to(device)
                logits1 = model(x)
                outputs = F.softmax(logits1, dim=1)
                softmax_outs.append(outputs)
        return torch.cat(softmax_outs, dim=0).to(device)
    else:
         loss_outs = torch.zeros(train_num).to(device)
         crit = nn.CrossEntropyLoss(reduction='none')
         with torch.no_grad():
            for x, y, idx in predict_loader:
                x = x.to(device)
                y = y.to(device)
                logits1 = model(x)
                loss = crit(logits1, y)
                loss_outs[idx] = loss
         return loss_outs