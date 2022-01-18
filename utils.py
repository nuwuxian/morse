import numpy as np
import torch
import datetime
import torch.nn.functional as F

def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


# Augmentation 
class MaskAug(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        len = x.shape[0]
        mask = np.random.binomial(1, p=self.ratio, size=len)
        masked_x = mask * x
        return np.float32(masked_x)

class NoiseAug(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        noise = np.random.rand(x.shape[0]) * self.ratio
        noise_x = x + noise
        return np.float32(noise_x)


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

def predict(predict_loader, model, device):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _ in predict_loader:
            if torch.cuda.is_available():
                images = images.to(device)
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model, device):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            logits1 = model(images1)
            logits2 = model(images2)
            outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            softmax_outs.append(outputs)
    return torch.cat(softmax_outs, dim=0).cpu()

def predict_dataset_softmax(predict_loader, model, device):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, _ in predict_loader:
            images1 = images1.to(device)
            logits1 = model(images1)
            outputs = F.softmax(logits1, dim=1)
            softmax_outs.append(outputs)
    return torch.cat(softmax_outs, dim=0).cpu()