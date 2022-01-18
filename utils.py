import numpy as np



def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


# Augmentation 
class MaskAug(object):
	def __init__(self, ratio):
		self.ratio = ratio

	def __call__(self, x):
        prob = torch.zeros_like(x).fill_(self.ratio)
        mask = torch.bernoulli(prob)
        masked_x = mask * x
        return masked_x


class NoiseAug(object):
	def __init__(self, ratio):
        self.ratio = ratio

	def __call__(self, x):
        noise = torch.randn(x.size()) * self.ratio
        noise_x = x + noise 
        return noise_x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

def predict(predict_loader, model):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _ in predict_loader:
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1 = model(images1)
                logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()