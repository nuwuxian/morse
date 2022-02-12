import os
import argparse
from dataset import get_dataset
from model import MLP_Net
import torch.backends.cudnn as cudnn
import torch
from our_match import our_match
import torch.nn as nn

from os import path as osp
from utils import make_timestamp
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4) # 1e-3 for malware-real
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=2e-4)

parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')

parser.add_argument('--input_dim', type=int, default=2381)

parser.add_argument('--gamma', type=float, default=0.95, metavar='M',help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'malware')

parser.add_argument('--epoch', type=int, default=140)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--cuda', type = int, default=1)
parser.add_argument('--num_class', type = int, default=10)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
parser.add_argument('--gpu_index', type=int, default=0)

# noise_setting | imbalanced setting
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument('--noise_type', type = str,  default='imb_step_0.1') # none for malware-real
parser.add_argument('--imb_type', type = str, default='step') # none for malware-real
parser.add_argument('--imb_ratio', type = float, default=0.1)

parser.add_argument('--lambda-u', default=1.0, type=float,
                        help='coefficient of unlabeled loss')

parser.add_argument('--T', default=0.5, type=float,
                        help='pseudo label temperature')

parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
# whether use the pretrain model
parser.add_argument('--use_pretrain', default=True, type=bool)
# divide data into clean / noise 
parser.add_argument('--clean_method', default='small_loss', type=str)
parser.add_argument('--clean_theta', default=0.95, type=float)
# imbalance method
parser.add_argument('--imb_method', default='reweight', type=str)   # resample / mixup / reweight
parser.add_argument('--reweight_start', default=50, type=int)
# mixup alpha
parser.add_argument('--alpha', default=10, type=int)
parser.add_argument('--use_true_distribution', default=False, type=bool)
parser.add_argument('--unlabel_reweight', default=True, type=bool)

parser.add_argument('--use_scl', default=False, type=bool)
parser.add_argument('--lambda-s', default=0.1, type=float)

# real-dataset | synthetic-dataset
parser.add_argument('--dataset_origin', default='synthetic', type=str) # real / synthetic

args = parser.parse_args()
root = './data/synthetic'
dataset = args.dataset
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

# Hyper Parameters
batch_size = args.batch_size
input_dim = args.input_dim

num_classes = args.num_class

train_dataset, test_dataset, train_data, noisy_targets, \
                clean_targets = get_dataset(root, args.dataset, args.noise_type, args.imb_type, args.imb_ratio)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

out_dir = 'output/Imb-method-' + str(args.imb_method) + '_reweight-start-' + str(args.reweight_start) \
            + '_lr-' + str(args.lr) + '_weight-decay-' + str(args.weight_decay)

timestamp = make_timestamp()
exp_name = args.seed
SAVE_DIR  = osp.join(out_dir, '{}-{}'.format(timestamp, exp_name))
os.makedirs(SAVE_DIR, exist_ok=True)

# check if gpu training is available
if torch.cuda.is_available():
   args.device = torch.device('cuda')
   cudnn.deterministic = True
   cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1

if args.dataset_origin == 'real':
   model = MLP_Net(input_dim, [512, 512, num_classes], batch_norm=nn.BatchNorm1d, use_scl=args.use_scl)
else:
   model = MLP_Net(input_dim, [1200, 1200, num_classes], batch_norm=nn.BatchNorm1d, use_scl=args.use_scl)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
# Cosin Learning Rates
# gamma 0.5 for real
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[10, 60], gamma=0.1, last_epoch=-1)

dist = [0.14, 0.15, 0.15, 0.12, 0.15, 0.09, 0.01, 0.12, 0.03, 0.02, 0.01, 0.01]

with torch.cuda.device(args.gpu_index):
    ourmatch = our_match(model=model, optimizer=optimizer, \
                         scheduler=lr_scheduler, logdir=SAVE_DIR, dist=dist, 
                         args=args)
    ourmatch.run(train_data, clean_targets, noisy_targets, \
                 train_loader, test_loader)
