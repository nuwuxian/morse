import numpy as np
from tqdm import tqdm
import timeit
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import AverageMeter, predict_dataset_softmax, get_labeled_dist
from utils import debug_label_info, debug_unlabel_info, debug_real_label_info, debug_real_unlabel_info
from utils import refine_pesudo_label, update_proto, init_prototype
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

        self.per_cls_weights = None
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
        
        soft_outs = predict_dataset_softmax(train_loader, self.model, self.args.device, self.train_num)

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

        per_cls_weights = None
        # update criterion
        if self.args.imb_method == 'reweight' and self.args.reweight_start != -1:
           cls_num_list = imb_labeled_sampler.label_to_count
           beta = 0.9999
           effective_num = 1.0 - np.power(beta, cls_num_list)
           per_cls_weights = (1.0 - beta) / np.array(effective_num)
           per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
           per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.args.device)
           for i in range(self.args.num_class):
               self.writer.add_scalar('Class' + str(i) + '_weight',  per_cls_weights[i], global_step=self.update_cnt)
        print('Labeled num is %d Unlabled num is %d' %(labeled_num, unlabeled_num))

        return labeled_dataset, labeled_loader,  unlabeled_loader, imb_labeled_loader, per_cls_weights


    def run(self, train_data, clean_targets, noisy_targets, trainloader, testloader):

        self.train_num = clean_targets.shape[0]
        best_acc = 0.0

        # dist_alignment or not
        if self.args.dist_alignment:
          self.model_dist = None

        for i in range(self.args.epoch):
            if i < self.args.warmup:
                self.warmup(i, trainloader)
                acc, class_acc = self.eval(testloader, self.model, i)
            else:
                # Labeled / UnLabeled DataLoader update only once
                if i == self.args.warmup:
                  start_time = timeit.default_timer()
                  labeled_dataset, labeled_loader, unlabeled_loader, imb_labeled_loader, per_cls_weights = \
                                   self.update_loader(trainloader, train_data, clean_targets, noisy_targets)
                  print('Prepare Data Loader Time: ', timeit.default_timer() - start_time)
                
                if i > self.args.reweight_start:
                  self.per_cls_weights = per_cls_weights

                if self.args.use_proto:
                  self.prototype = init_prototype(labeled_loader, self.model, self.args.device, self.args.num_class)

                start_time = timeit.default_timer()
                self.ourmatch_train(i, labeled_dataset, labeled_loader, unlabeled_loader, imb_labeled_loader)
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

    def ourmatch_train(self, epoch, labeled_dataset, labeled_loader, unlabeled_loader, imb_labeled_loader):

        self.model.train()

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_s = AverageMeter()
        # labeled distribution
        labeled_dist = get_labeled_dist(labeled_dataset).to(self.args.device)

        debug_list = [AverageMeter() for _ in range(self.args.num_class * 2)]

        for batch_idx, (b_l, b_u, b_imb_l) in enumerate(zip(labeled_loader, unlabeled_loader, imb_labeled_loader)):
                
                # unpack b_l, b_u, b_imb_l
                xl, yl = b_l
                xw, xs, gts_u = b_u
                x_imb_l, y_imb_l = b_imb_l
                # shift
                xl, xw, xs, x_imb_l = xl.to(self.args.device), xw.to(self.args.device), \
                                      xs.to(self.args.device), x_imb_l.to(self.args.device)

                yl, y_imb_l = yl.to(self.args.device), y_imb_l.to(self.args.device)

                logits_xw = self.model(xw)
                logits_xs = self.model(xs)

                if self.args.imb_method == 'resample':
                   logits_imb_x = self.model(x_imb_l)
                   Lx = F.cross_entropy(logits_imb_x, y_imb_l, reduction='mean')

                elif self.args.imb_method == 'mixup':
                   lam = np.random.beta(self.args.alpha, self.args.alpha)
                   lam = max(lam, 1 - lam)
                   idx = torch.randperm(xl.size()[0])
                   # mixup in hidden-layers
                   mix_x = self.model.forward_encoder(x_imb_l) * lam + \
                            (1 - lam) * self.model.forward_encoder(xl)
                   mix_logits = self.model.forward_classifier(mix_x)
                   Lx = self.mixup_criterion(self.criterion, mix_logits, y_imb_l, yl[idx], lam)

                elif self.args.imb_method == 'reweight':
                  logits_x = self.model(xl)
                  Lx = F.cross_entropy(logits_x, yl, weight=self.per_cls_weights, reduction='mean')

                # Use Suploss
                if self.args.use_scl:
                    feat = self.model.forward_feat(xl)
                    feat = feat.unsqueeze(1)
                    Ls = self.criterion_con(feat, yl)

                with torch.no_grad():
                    probs = torch.softmax(logits_xw.detach() / self.args.T, -1)
                    if self.args.dist_alignment:
                      if self.model_dist == None:
                        self.model_dist = torch.mean(probs, dim=0)
                      else:
                        self.model_dist = self.model_dist * 0.999 + torch.mean(probs, dim=0) * 0.001
                      probs *= (labeled_dist + self.args.dist_alignment_eps) / (self.model_dist + self.args.dist_alignment_eps)
                      probs /= probs.sum(-1, keepdim=True)

                    if not self.args.use_proto:
                       yu = torch.argmax(probs, -1)
                       mask = (torch.max(probs, -1)[0] >= self.threshold).to(dtype=torch.float32)
                    else:
                       yu, mask = refine_pesudo_label(xw, probs, self.threshold, self.prototype, self.model)

                debug_ratio = debug_unlabel_info(yu, gts_u, mask, self.args.num_class)
                for i in range(len(debug_list)):
                  debug_list[i].update(debug_ratio[i])
                
                Lu = (F.cross_entropy(logits_xs, yu, weight=self.per_cls_weights, reduction='none') * mask).mean()
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
                if self.args.use_proto:
                   self.prototype = update_proto(xl, yl, self.prototype, self.model)

        print('Epoch [%3d/%3d] \t Losses: %.8f, Losses_x: %.8f Losses_u: %.8f'% (epoch, self.args.epoch, losses.avg, losses_x.avg, losses_u.avg))
        # write into tensorboard
        self.writer.add_scalar('Loss', losses.avg, self.update_cnt)
        self.writer.add_scalar('Loss_x', losses_x.avg, self.update_cnt)
        self.writer.add_scalar('Loss_u', losses_u.avg, self.update_cnt)
        if self.args.use_scl:
           self.writer.add_scalar('Loss_s', losses_s.avg, self.update_cnt)
        
        for i in range(2 * self.args.num_class):
            # debug info
            if i < self.args.num_class:
               self.writer.add_scalar('UnLabel_class' + str(i) + '-clean_ratio', debug_list[i].avg, self.update_cnt)
            else:
               cls = i % self.args.num_class
               self.writer.add_scalar('UnLabel_class' + str(cls) + '-prob', debug_list[i].avg, self.update_cnt)


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
        
        self.writer.add_scalar('Test Acc',  acc, epoch)

        return acc, class_acc