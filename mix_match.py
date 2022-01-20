import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cat
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.metrics import confusion_matrix

from utils import AverageMeter, predict_softmax, predict_dataset_softmax
from dataset import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset
from torch.utils.data import DataLoader


class mix_match(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # load model, optimizer, scheduler
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        self.threshold = self.args.threshold
        # tensorboard writer
        self.writer = SummaryWriter(log_dir=kwargs['log_dir'])
        self.update_cnt = 0

    def aug(self, input, aug_type):
        mask_ratio = []
        if aug_type == 'weak':
           mask_ratio.append(0.9)
        else:
           mask_ratio.append(0.9)  # weak augmentation
           mask_ratio.append(0.8)  # strong augmentation
        ret = []
        for i in range(len(mask_ratio)):
            ratio = mask_ratio[i]
            prob = torch.zeros_like(input).fill_(ratio)
            m = torch.bernoulli(prob)
            no, dim = input.shape
            # Randomly (and column-wise) shuffle data
            x_bar = np.zeros([no, dim])
            for i in range(dim):
                idx = np.random.permutation(no)
                x_bar[:, i] = input[idx, i]
            x_bar = torch.Tensor(x_bar)

            # Corrupt samples
            x_tilde = input * m + x_bar * (1 - m)
            ret.append(x_tilde)

        if aug_type == 'weak': return ret[0]
        if aug_type == 'weak_strong': return ret[0], ret[1]

        # mix-up
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def splite_confident(self, outs, clean_targets, noisy_targets):
        probs, preds = torch.max(outs.data, 1)

        labeled_indexs = []
        unlabeled_indexs = []

        for i in range(0, len(noisy_targets)):
            if preds[i] == noisy_targets[i] and probs[i] > 0.95:
                labeled_indexs.append(i)
            else:
                unlabeled_indexs.append(i)

        return labeled_indexs, unlabeled_indexs

    def update_trainloader(self, train_loader, train_data, clean_targets, noisy_targets):

        soft_outs = predict_dataset_softmax(train_loader, self.model, self.args.device)

        labeled_indexs, unlabeled_indexs = self.splite_confident(soft_outs, clean_targets, noisy_targets)
        labeled_dataset = Semi_Labeled_Dataset(train_data[labeled_indexs], noisy_targets[unlabeled_indexs])
        unlabeled_dataset = Semi_Unlabeled_Dataset(train_data[unlabeled_indexs])

        labeled_num, unlabeled_num = len(labeled_indexs), len(unlabeled_indexs)

        # print confident set clean-ratio
        clean_num = np.sum(noisy_targets[labeled_indexs]==clean_targets[labeled_indexs])
        clean_ratio = clean_num * 1.0 / labeled_num

        self.writer.add_scalar('Labeled_clean_ratio', clean_ratio, global_step=self.update_cnt)
        self.writer.add_scalar('Labeled_num', labeled_num, global_step=self.update_cnt)

        l_batch = self.args.batch_size
        u_batch = int(self.args.batch_size * min(6,  unlabeled_num * 1.0 / labeled_num))

        labeled_trainloader = DataLoader(dataset=labeled_dataset, batch_size=l_batch, shuffle=True, \
                                         num_workers=8, pin_memory=True, drop_last=True)

        unlabeled_trainloader = DataLoader(dataset=unlabeled_dataset, batch_size=u_batch, shuffle=True, \
                                           num_workers=8, pin_memory=True, drop_last=True)

        return labeled_trainloader, unlabeled_trainloader

    def run(self, train_data, clean_targets, noisy_targets, trainloader, testloader):

        for i in range(self.args.epoch):
            if i < self.args.warmup:
                self.warmup(i, trainloader)
            else:
                
                labeled_trainloader, unlabeled_trainloader = self.update_trainloader(trainloader, train_data, clean_targets, noisy_targets)
                self.fixmatch_train(i, labeled_trainloader, unlabeled_trainloader)

                self.update_cnt += 1

            self.optimizer.step()
            
            self.eval(testloader, i)

    def fixmatch_train(self, epoch, labeled_trainloader, unlabeled_trainloader):

        self.model.train()

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        for batch_idx, (b_l, b_u) in enumerate(zip(labeled_trainloader, unlabeled_trainloader)):
            # unpack b_l, b_u
            inputs_x, targets_x = b_l
            inputs_u = bu

            inputs_x, inputs_u = inputs_x.to(self.args.device), inputs_u.to(self.args.device)
            inputs_x = self.aug(inputs_x, 'weak')
            inputs_u, inputs_u2 = self.aug(inputs_u, 'weak_strong')

            targets_x = targets_x.to(self.args.device)

            logits_x = self.model(inputs_x)
            logits_u_w = self.model(inputs_u)
            logits_u_s = self.model(inputs_u2)

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + self.args.lambda_u * Lu
            # update model
            self.optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            self.optimizer.step()

        print('Epoch [%3d/%3d] \t Losses: %.8f, Losses_x: %.8f Losses_u: %.8f'% (epoch, self.args.epoch, losses.avg, losses_x.avg, losses_u.avg))
        # write into tensorboard
        self.writer.add_scalar('Loss', losses.avg, self.update_cnt)
        self.writer.add_scalar('Loss_x', losses_x.avg, self.update_cnt)
        self.writer.add_scalar('Loss_x', losses_u.avg, self.update_cnt)


    def warmup(self, epoch, trainloader):
        self.model.train()
        batch_idx = 0
        num_iter = (len(trainloader.dataset) // self.args.batch_size) + 1

        losses = AverageMeter()

        for i, (x, y) in enumerate(trainloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            logits = self.model(x)
            
            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), len(logits))
            batch_idx += 1

        print('Epoch [%3d/%3d] Loss: %.2f' % (epoch, self.args.epoch, losses.avg))

    def eval(self, testloader, epoch):
        self.model.eval()  # Change model to 'eval' mode.
        correct = 0
        total = 0

        # return the class-level accuracy
        model_preds = []
        model_true = []

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                logits = self.model(x)

                outputs = F.softmax(logits, dim=1)
                _, pred = torch.max(outputs.data, 1)

                total += y.size(0)
                correct += (pred.cpu() == y.long()).sum()

                # add pred1 | labels
                model_preds.append(pred.cpu())
                model_true.append(y.long())

        model_preds = np.concatenate(model_preds, axis=0)
        model_true = np.concatenate(model_true, axis=0)

        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, self.args.epoch, acc))
        self.writer.add_scalar('Test Acc',  acc, epoch)