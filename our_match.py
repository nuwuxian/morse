import numpy as np
from tqdm import tqdm
import timeit
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import AverageMeter, predict_dataset_softmax
from utils import debug_label_info, debug_unlabel_info, debug_real_label_info, debug_real_unlabel_info
from dataset import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset,  ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from losses import LDAMLoss, SupConLoss

class our_match(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # load model, ema_model, optimizer, scheduler
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        self.threshold = self.args.threshold
        self.log_dir = kwargs['logdir']
        # tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.update_cnt = 0
        # Distribution of noisy data
        self.dist = kwargs['dist']
        self.cls_threshold = np.array(self.dist) * self.args.threshold / max(self.dist)
        self.cls_threshold = torch.Tensor(self.cls_threshold).to(self.args.device) 

        if self.args.imb_method == 'resample' or self.args.imb_method == 'mixup':
           self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        # SupConLoss
        self.criterion_con = SupConLoss(temperature=0.07)

        # mix-up
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def splite_confident(self, outs, clean_targets, noisy_targets):
        labeled_indexs = []
        unlabeled_indexs = []

        if not self.args.use_true_distribution:
            if self.args.clean_method == 'confidence':
                probs, preds = torch.max(outs.data, 1)
                for i in range(0, len(noisy_targets)):
                    if preds[i] == noisy_targets[i] and probs[i] > self.args.clean_theta:
                        labeled_indexs.append(i)
                    else:
                        unlabeled_indexs.append(i)
            elif self.args.clean_method == 'small_loss':
                 for cls in range(self.args.num_class):
                     idx = np.where(noisy_targets==cls)[0]
                     loss_cls = outs[idx]
                     sorted, indices = torch.sort(loss_cls, descending=False)
                     select_num = int(len(indices) * 0.15)
                     for i in range(len(indices)):
                         if i < select_num:
                            labeled_indexs.append(idx[indices[i].item()])
                         else:
                            unlabeled_indexs.append(idx[indices[i].item()])
        else:
            sz = noisy_targets.shape[0]
            for cls in range(self.args.num_class):
                idx = np.where(noisy_targets == cls)[0]
                select_num = int(sz * 0.15 * self.dist[cls])
                cnt = 0
                for i in range(len(idx)):
                    if noisy_targets[idx[i]] == clean_targets[idx[i]]:
                       cnt += 1
                       if cnt <= select_num:
                          labeled_indexs.append(idx[i])
                       else:
                          unlabeled_indexs.append(idx[i])
                    else:
                         unlabeled_indexs.append(idx[i])

        return labeled_indexs, unlabeled_indexs

    def update_loader(self, train_loader, train_data, clean_targets, noisy_targets):
        
        soft_outs = predict_dataset_softmax(train_loader, self.model, self.args.device, self.train_num, self.args.clean_method)

        labeled_indexs, unlabeled_indexs = self.splite_confident(soft_outs, clean_targets, noisy_targets)
        labeled_dataset = Semi_Labeled_Dataset(train_data[labeled_indexs], noisy_targets[labeled_indexs])
        unlabeled_dataset = Semi_Unlabeled_Dataset(train_data[unlabeled_indexs], clean_targets[unlabeled_indexs])

        labeled_num, unlabeled_num = len(labeled_indexs), len(unlabeled_indexs)

        # print confident set clean-ratio
        clean_num = np.sum(noisy_targets[labeled_indexs]==clean_targets[labeled_indexs])
        clean_ratio = clean_num * 1.0 / labeled_num

        noise_label = noisy_targets[labeled_indexs]
        clean_label = clean_targets[labeled_indexs]
       
        print('Labeled data clean ratio is %.2f' %clean_ratio)

        cls_precision = debug_label_info(noise_label, clean_label, self.args.num_class)
        
        att_cls = range(self.args.num_class)
        
        for idx in range(len(att_cls)):
            cls = att_cls[idx]
            self.writer.add_scalar('Label_class' + str(cls) + '-clean_ratio', cls_precision[cls], global_step=self.update_cnt)
        
        self.writer.add_scalar('Label_clean_ratio', clean_ratio, global_step=self.update_cnt)    
        self.writer.add_scalar('Label_num', labeled_num, global_step=self.update_cnt)

        l_batch = self.args.batch_size
        u_batch = self.args.batch_size * 6
        # labeled_loader / unlabeled_loader
        labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=l_batch, shuffle=True,
                                    num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
        unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=u_batch, shuffle=True,
                                      num_workers=self.args.num_workers, pin_memory=True,
                                      drop_last=True)
        # imb_labeled_loader
        imb_labeled_sampler =  ImbalancedDatasetSampler(labeled_dataset, num_class=self.args.num_class)
        imb_labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=l_batch, shuffle=False,
                             num_workers=self.args.num_workers, pin_memory=True, sampler=imb_labeled_sampler,
                             drop_last=True)
        self.per_cls_weights = None
        # update criterion
        if self.args.imb_method == 'reweight' and self.args.reweight_start != -1:
           cls_num_list = imb_labeled_sampler.label_to_count
           if self.update_cnt >= self.args.reweight_start:
              beta = 0.9999
           else:
              beta = 0
           effective_num = 1.0 - np.power(beta, cls_num_list)
           per_cls_weights = (1.0 - beta) / np.array(effective_num)
           per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
           per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.args.device)
           self.per_cls_weights = per_cls_weights
           for i in range(self.args.num_class):
               self.writer.add_scalar('Class' + str(i) + '_weight', self.per_cls_weights[i], global_step=self.update_cnt)
        print('Labeled num is %d Unlabled num is %d' %(labeled_num, unlabeled_num))

        return labeled_loader,  unlabeled_loader, imb_labeled_loader

    def run(self, train_data, clean_targets, noisy_targets, trainloader, testloader):
        self.train_num = clean_targets.shape[0]
        best_acc = 0.0
        for i in range(self.args.epoch):
            if i < self.args.warmup:
                self.warmup(i, trainloader)
                acc, class_acc = self.eval(testloader, self.model, i)

            else:
                start_time = timeit.default_timer()
                labeled_loader, unlabeled_loader, imb_labeled_loader = self.update_loader(trainloader, train_data, clean_targets, noisy_targets)
                print('Prepare Data Loader Time: ', timeit.default_timer() - start_time)
                if i == self.args.warmup and not self.args.use_pretrain:
                    self.model.init()

                start_time = timeit.default_timer()
                self.ourmatch_train(i, labeled_loader, unlabeled_loader, imb_labeled_loader)
                print('Training Time: ', timeit.default_timer() - start_time)

                self.update_cnt += 1
                start_time = timeit.default_timer()
                acc, class_acc = self.eval(testloader, self.model, i)
                print('Eval Time: ', timeit.default_timer() - start_time)
                if acc > best_acc:
                    best_acc = acc
                    np.savez_compressed(self.log_dir + '/best_results.npz', test_acc=best_acc, test_class_acc=class_acc,
                                        best_epoch=i)
            self.scheduler.step()

    def ourmatch_train(self, epoch, labeled_loader, unlabeled_loader, imb_labeled_loader):

        self.model.train()

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_s = AverageMeter()

        # 0 -> 0, 1, ... 11
        # 11 -> 11
        if self.args.dataset_origin == 'real':
           debug_list = [AverageMeter() for _ in range(self.args.num_class+1)]
        else:
           debug_list = [AverageMeter() for _ in range(self.args.num_class)]

        for batch_idx, (b_l, b_u, b_imb_l) in enumerate(zip(labeled_loader, unlabeled_loader, imb_labeled_loader)):
                
                # unpack b_l, b_u, b_imb_l
                inputs_x, targets_x = b_l
                inputs_u, inputs_u2, gts_u = b_u
                inputs_imb_x, targets_imb_x = b_imb_l

                # shifit to gpu/cpu
                inputs_x, inputs_u, inputs_u2, inputs_imb_x = inputs_x.to(self.args.device), inputs_u.to(self.args.device), \
                                                              inputs_u2.to(self.args.device), inputs_imb_x.to(self.args.device)
                targets_x, targets_u, targets_imb_x = targets_x.to(self.args.device), \
                                                      targets_x.to(self.args.device), targets_imb_x.to(self.args.device)

                logits_u_w = self.model(inputs_u)
                logits_u_s = self.model(inputs_u2)

                if self.args.imb_method == 'resample':
                   logits_imb_x = self.model(inputs_imb_x)
                   Lx = F.cross_entropy(logits_imb_x, targets_imb_x, reduction='mean')

                elif self.args.imb_method == 'mixup':
                   lam = np.random.beta(self.args.alpha, self.args.alpha)
                   lam = max(lam, 1 - lam)
                   idx = torch.randperm(inputs_x.size()[0])
                   # mixup in hidden-layers
                   mix_x = self.model.forward_encoder(inputs_imb_x) * lam + \
                            (1 - lam) * self.model.forward_encoder(inputs_x)
                   mix_logits = self.model.forward_classifier(mix_x)
                   Lx = self.mixup_criterion(self.criterion, mix_logits, targets_imb_x, targets_x[idx], lam)

                elif self.args.imb_method == 'reweight':
                  logits_x = self.model(inputs_x)
                  Lx = F.cross_entropy(logits_x, targets_x, weight=self.per_cls_weights, reduction='mean')

                # Use Suploss
                if self.args.use_scl:
                    feat = self.model.forward_feat(inputs_x)
                    feat = feat.unsqueeze(1)
                    Ls = self.criterion_con(feat, targets_x)

                pseudo_label = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                if self.args.use_pretrain:
                   mask = max_probs.ge(self.args.threshold).float().to(self.args.device)
                else:
                   # class specific threshold
                   mask = max_probs.ge(self.cls_threshold[targets_u]).float().to(self.args.device)

               
                debug_ratio = debug_unlabel_info(targets_u, gts_u, mask, self.args.num_class)
                for i in range(len(debug_list)):
                  debug_list[i].update(debug_ratio[i])
                
                Lu = (F.cross_entropy(logits_u_s, targets_u, weight=self.per_cls_weights, reduction='none') * mask).mean()
                if self.args.use_scl:
                   loss = Lx + self.args.lambda_u * Lu + self.args.lambda_s * Ls
                else:
                   loss = Lx + self.args.lambda_u * Lu
                # update model
                self.optimizer.zero_grad()
                loss.backward()
                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                if self.args.use_scl:
                    losses_s.update(Ls.item())

                self.optimizer.step()

        print('Epoch [%3d/%3d] \t Losses: %.8f, Losses_x: %.8f Losses_u: %.8f'% (epoch, self.args.epoch, losses.avg, losses_x.avg, losses_u.avg))
        # write into tensorboard
        self.writer.add_scalar('Loss', losses.avg, self.update_cnt)
        self.writer.add_scalar('Loss_x', losses_x.avg, self.update_cnt)
        self.writer.add_scalar('Loss_u', losses_u.avg, self.update_cnt)
        if self.args.use_scl:
           self.writer.add_scalar('Loss_s', losses_s.avg, self.update_cnt)
        
        for i in range(self.args.num_class):
            # debug info
            self.writer.add_scalar('UnLabel_class' + str(i) + '-clean_ratio', debug_list[i].avg, self.update_cnt)


    def warmup(self, epoch, trainloader):
        self.model.train()
        batch_idx = 0
        losses = AverageMeter()

        for i, (x, y, _) in enumerate(trainloader):
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

    def eval(self, testloader, eval_model, epoch):
        eval_model.eval()  # Change model to 'eval' mode.
        correct = 0
        total = 0

        # return the class-level accuracy
        model_preds = []
        model_true = []

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                logits = eval_model(x)

                outputs = F.softmax(logits, dim=1)
                _, pred = torch.max(outputs.data, 1)

                total += y.size(0)
                correct += (pred.cpu() == y.cpu().long()).sum()

                # add pred1 | labels
                model_preds.append(pred.cpu())
                model_true.append(y.cpu().long())

        model_preds = np.concatenate(model_preds, axis=0)
        model_true = np.concatenate(model_true, axis=0)

        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        print(class_acc)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, self.args.epoch, acc))

        if self.args.dataset_origin != 'real':
           print('Large Class Accuracy is %.2f Small Class Accuracy is %.2f' %(np.mean(class_acc[:5]), np.mean(class_acc[5:])))
           self.writer.add_scalar('Large Class acc', np.mean(class_acc[:5]), epoch)
           self.writer.add_scalar('Small Class acc', np.mean(class_acc[5:]), epoch)
           for i in range(self.args.num_class):
               self.writer.add_scalar('Test Class-' + str(i) + ' acc', class_acc[i], epoch)
        else:
           self.writer.add_scalar('Test Class-0 acc', class_acc[0], epoch)
           self.writer.add_scalar('Test Class-11 acc', class_acc[self.args.num_class-1], epoch)

        self.writer.add_scalar('Test Acc',  acc, epoch)

        return acc, class_acc