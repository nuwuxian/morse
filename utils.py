import numpy as np
import torch
import datetime
import torch.nn.functional as F
import time
import torch.nn as nn
def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)

# cosdistance
def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix


def count_knn_distribution(args, feat_cord, label, cluster_sum, k, norm='l2'):
    # feat_cord = torch.tensor(final_feat)
    KINDS = args.num_class
    
    dist = cosDistance(feat_cord)

    print(f'knn parameter is k = {k}')
    time1 = time.time()
    min_similarity = 0
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]

    knn_labels = label[indices]


    knn_labels_cnt = torch.zeros(cluster_sum, KINDS)

    for i in range(KINDS):
        knn_labels_cnt[:, i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i), 1)

        # print(knn_labels_cnt[0])
    time2 = time.time()
    print(f'Running time for k = {k} is {time2 - time1}')

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob


def get_knn_acc_all_class(args, data_set, k=10, sel_noisy=None):
    # Build Feature Clusters --------------------------------------
    KINDS = args.num_class

    all_point_cnt = data_set['feature'].shape[0]
    # global
    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    sel_idx = data_set['index'][sample]
    knn_labels_cnt = count_knn_distribution(args, final_feat, noisy_label, all_point_cnt, k=k, norm='l2')
    # test majority voting
    print(f'Use MV')
    label_pred = np.argmax(knn_labels_cnt, axis=1).reshape(-1)
    sel_noisy += (sel_idx[label_pred != noisy_label]).tolist()
    
    return sel_noisy



def noniterate_detection(args, data_set, train_dataset, sel_noisy=[]):
    # non-iterate
    # sel_noisy = []
    # print(data_set['noisy_label'])
    sel_noisy = get_knn_acc_all_class(args, data_set, k=args.k, sel_noisy=sel_noisy)

    sel_noisy = np.array(sel_noisy)
    sel_clean = np.array(list(set(data_set['index'].tolist()) ^ set(sel_noisy)))

    noisy_in_sel_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / sel_noisy.shape[0]
    precision_noisy = noisy_in_sel_noisy
    recall_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / np.sum(train_dataset.noise_or_not)

    noisy_in_sel_clean = np.sum(train_dataset.noise_or_not[sel_clean]) / sel_clean.shape[0]
    print(f'[noisy] precision: {precision_noisy}')
    print(f'[noisy] recall: {recall_noisy}')
    print(f'[noisy] F1-score: {2.0 * precision_noisy * recall_noisy / (precision_noisy + recall_noisy)}')

    import pdb; pdb.set_trace()

    return sel_noisy, sel_clean, data_set['index']

 


def noise_detect(model, train_loader, train_dataset, args):
    model.eval()

    feats, labels, indexs = [], [], []
    with torch.no_grad():
        for i_batch, (feature, label, index) in enumerate(train_loader):
            feature = feature.to(args.device)
            label = label.to(args.device)
            extracted_feature = model.forward_encoder(feature)

            # feat / label / index
            feats.append(extracted_feature.detach().cpu())
            labels.append(label.detach().cpu())
            indexs.append(index)


    # concat feat, label, index
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    indexs = torch.cat(indexs, 0)

    dataset = {'feature': feats, 'noisy_label': labels, 'index': indexs}
    noniterate_detection(args, dataset, train_dataset, [])

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

def predict_dataset_softmax(predict_loader, model, train_num, device):
    # model.eval()
    # softmax_outs = []
    # with torch.no_grad():
    #     for images1, _, _ in predict_loader:
    #         images1 = images1.to(device)
    #         logits1 = model(images1)
    #         outputs = F.softmax(logits1, dim=1)
    #         softmax_outs.append(outputs)
    # return torch.cat(softmax_outs, dim=0).cpu()

    model.eval()
    loss_outs = torch.zeros(train_num).to(device)

    with torch.no_grad():
        for images1, label, idx in predict_loader:
            images1 = images1.to(device)
            logits1 = model(images1)
            # outputs = F.softmax(logits1, dim=1)
            loss = nn.CrossEntropyLoss(reduce=False)(logits1, label)
            loss_outs[idx] = loss
    return loss_outs
