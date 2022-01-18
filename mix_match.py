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


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


class mix_match(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # load model, optimizer, scheduler
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        self.threshold = self.args.theta
        self.labeled_dist = torch.Tensor(kwargs['labeled_dist']).to(self.args.device)
        # define class_theta
        self.class_threshold = self.labeled_dist * self.threshold / max(self.labeled_dist)
        self.class_threshold = torch.Tensor(self.class_threshold).to(self.args.device)


    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    # mix-up
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def splite_confident(self, outs, clean_targets, noisy_targets):
        probs, preds = torch.max(outs.data, 1)

        confident_correct_num = 0
        unconfident_correct_num = 0
        confident_indexs = []
        unconfident_indexs = []

        for i in range(0, len(noisy_targets)):
            if preds[i] == noisy_targets[i] and probs[i] > 0.95:
                confident_indexs.append(i)
                if clean_targets[i] == preds[i]:
                    confident_correct_num += 1
            else:
                unconfident_indexs.append(i)
                if clean_targets[i] == preds[i]:
                    unconfident_correct_num += 1

        return confident_indexs, unconfident_indexs


    def update_trainloader(self, train_loader, train_data, clean_targets, noisy_targets, transform_train):

        # predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
        # predict_loader =torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=self.args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        #
        # soft_outs = predict_softmax(predict_loader, self.model, self.args.device)
        soft_outs = predict_dataset_softmax(train_loader, self.model, self.args.device)

        confident_indexs, unconfident_indexs = self.splite_confident(soft_outs, clean_targets, noisy_targets)
        confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
        unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], transform_train)

        # print confident set clean-ratio
        clean_ratio = np.sum(noisy_targets[confident_indexs] == clean_targets[confident_indexs]) * 1.0 / len(confident_indexs)
        print('clean ratio is %f' %clean_ratio)


        # uncon_batch = int(self.args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * self.args.batch_size)
        # con_batch = self.args.batch_size - uncon_batch
        print('confident sz is %d unconfident sz is %d' %(len(confident_indexs), len(unconfident_indexs)))
        uncon_batch = con_batch = self.args.batch_size

        labeled_trainloader = torch.utils.data.DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        unlabeled_trainloader = torch.utils.data.DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

        return labeled_trainloader, unlabeled_trainloader

    def run(self, train_data, clean_targets, noisy_targets, transform_train, trainloader, testloader):

        for i in range(self.args.epoch):
            if i < self.args.warmup:
                self.warmup(i, trainloader)
            else:
                labeled_trainloader, unlabeled_trainloader = self.update_trainloader(trainloader, train_data, clean_targets, noisy_targets, transform_train)
                self.fixmatch_train(i, labeled_trainloader, unlabeled_trainloader)
            self.optimizer.step()
            self.eval(testloader, i)


    def fixmatch_train(self, epoch, labeled_trainloader, unlabeled_trainloader):

        self.model.train()
        unlabeled_train_iter = iter(unlabeled_trainloader)    
        num_iter = (len(labeled_trainloader.dataset)//self.args.batch_size)+1

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        for batch_idx, (inputs_x, inputs_x2, targets_x) in enumerate(labeled_trainloader):

            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()


            inputs_x, inputs_x2 = inputs_x.to(self.args.device), inputs_x2.to(self.args.device)
            targets_x = targets_x.to(self.args.device)
            inputs_u, inputs_u2 = inputs_u.to(self.args.device), inputs_u2.to(self.args.device)

            batch_size = inputs_x.size(0)

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
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            self.optimizer.step()

            self.model.zero_grad()
        print('Epoch [%3d/%3d] \t Losses: %.8f, Losses_x: %.8f Losses_u: %.8f'% (epoch, self.args.epoch, losses.avg, losses_x.avg, losses_u.avg))


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