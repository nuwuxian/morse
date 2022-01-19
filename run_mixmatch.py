import os
import torchvision.transforms as transforms
import argparse
import numpy as np
import datetime
from dataset import get_dataset
from model import MLP_Net
import torch.backends.cudnn as cudnn
import torch
from mix_match import mix_match
import torch.nn as nn

from os import path as osp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import math
from utils import MaskAug, NoiseAug


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

parser = argparse.ArgumentParser()


parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=2e-4)
parser.add_argument('--input_dim', type=int, default=1024)

parser.add_argument('--gamma', type=float, default=0.95, metavar='M',help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'malware')

parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--cuda', type = int, default=1)
parser.add_argument('--num_class', type = int, default=12)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
parser.add_argument('--gpu_index', type=int, default=0)


# semi-superivsed setting
parser.add_argument('--mu', default=1.0, type=int,
                        help='coefficient of unlabeled batch size')

parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')

parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')

parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')

parser.add_argument('--theta', default=0.95, type=float,
                        help='pseudo label threshold')

args = parser.parse_args()
root = './data'
dataset = args.dataset
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr
input_dim = args.input_dim

num_classes = args.num_class

train_dataset, test_dataset, train_data, noisy_targets, \
                clean_targets = get_dataset(root, args.dataset)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)


model = MLP_Net(input_dim, [512, 512, num_classes], batch_norm=nn.BatchNorm1d)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[30, 60], gamma=0.9, last_epoch=-1)

# check if gpu training is available
if torch.cuda.is_available():
   args.device = torch.device('cuda')
   cudnn.deterministic = True
   cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1


# suppose label distribution is given
labeled_dist = [0.14, 0.15, 0.15, 0.12, 0.15, 0.09, 0.01, 0.12, 0.03, 0.02, 0.01, 0.01]
labeled_dist = np.array(labeled_dist)

with torch.cuda.device(args.gpu_index):
    mixmatch = mix_match(model=model, optimizer=optimizer, scheduler=lr_scheduler,
                         args=args, labeled_dist=labeled_dist)
    mixmatch.run(train_data, clean_targets, noisy_targets, \
                 train_loader, test_loader)