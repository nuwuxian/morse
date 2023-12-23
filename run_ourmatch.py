import os
import argparse
from dataset import get_dataset
from model import MLP_Net
import torch.backends.cudnn as cudnn
import torch
from our_match import our_match
import torch.nn as nn
import json
from os import path as osp
from utils import make_timestamp
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-3) 
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=2e-4)

parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')

parser.add_argument('--input_dim', type=int, default=2381) # 1024 for malware-real, 2381 for malware-syn

parser.add_argument('--gamma', type=float, default=0.95, metavar='M',help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'malware')

parser.add_argument('--epoch', type=int, default=100) # 140 for malware-real, 100 for malware-syn
parser.add_argument('--warmup', type=int, default=5) # 5 for malware-real, 10 for malware-syn
parser.add_argument('--optimizer', type = str, default='adam')  
parser.add_argument('--cuda', type = int, default=0)
parser.add_argument('--num_class', type = int, default=10) # 12 for malware-real, 10 for malware-syn

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--gpu_index', type=int, default=0) 

# noise_setting | imbalanced setting
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, shouxld be less than 1', default = 0.6)
parser.add_argument('--noise_type', type = str,  default='imb_step_0.05') # none for malware-real
parser.add_argument('--imb_type', type = str, default='step') # none for malware-real, step for malware-syn
parser.add_argument('--imb_ratio', type = float, default=0.05)

parser.add_argument('--lambda-u', default=1.0, type=float,
                        help='coefficient of unlabeled loss')

parser.add_argument('--T', default=0.5, type=float,
                        help='pseudo label temperature') # higer temperature, probs are closer

parser.add_argument('--threshold', default=0.40, type=float, # 0.95 for malware-real, 0.40 for malware-syn
                        help='pseudo label threshold')
# whether use the pretrain model
parser.add_argument('--use_pretrain', default=True, type=bool)
# divide data into clean / noise 
parser.add_argument('--clean_method', default='confidence', type=str)
parser.add_argument('--clean_theta', default=0.95, type=float)
# imbalance method
parser.add_argument('--imb_method', default='reweight', type=str)   # default is 're-weight'
parser.add_argument('--reweight_start', default=20, type=int) # 40 for malware-real
# mixup alpha
parser.add_argument('--alpha', default=10, type=int)
parser.add_argument('--use_true_distribution', default=False, type=bool)
parser.add_argument('--unlabel_reweight', default=True, type=bool)

parser.add_argument('--dist_alignment', default=False, type=bool)
parser.add_argument('--dist_alignment_eps', default=1e-6, type=float)
parser.add_argument('--dist_alignment_batches', default=5, type=int)

parser.add_argument('--use_scl', default=False, type=bool)
parser.add_argument('--lambda-s', default=0.1, type=float)

parser.add_argument('--use_proto', default=False, type=bool)
parser.add_argument('--use_hard_labels', default=False, type=bool) # soft version is better than hard label
parser.add_argument('--use_dynamic_threshold', default=False, type=bool)
parser.add_argument('--epsilon', default=0.95, type=float) # 0.7 for malware-real

# real-dataset | synthetic-dataset
parser.add_argument('--dataset_origin', default='synthetic', type=str) # real / synthetic

args = parser.parse_args()


root = './data/real_world'
if args.dataset_origin != 'real':
   root = './data/synthetic'

dataset = args.dataset
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

# Hyper Parameters
batch_size = args.batch_size
input_dim = args.input_dim

num_classes = args.num_class

train_dataset, test_dataset, train_data, \
noisy_targets, clean_targets = get_dataset(root, args.dataset, args.noise_type, args.noise_rate, \
                               args.imb_type, args.imb_ratio, args.num_class)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

sub_fold = 'real/'
if args.dataset_origin != 'real':
    sub_fold = 'syn/'

out_dir = 'tmp/' + sub_fold + 'imb-method_' + str(args.imb_method) + '_noise-type_' + str(args.noise_type) + '_noise-rate_' + str(args.noise_rate) \
                + '_imb-type_' + str(args.imb_type) + '_imb-rate_' + str(args.imb_ratio) + '_reweight-start-' \
                + str(args.reweight_start) + '_dist-alignment-' + str(args.dist_alignment) + '_use-hard-labels_' \
                + str(args.use_hard_labels) + '_ratio_' + str(args.epsilon)

timestamp = make_timestamp()
exp_name = args.seed
SAVE_DIR  = osp.join(out_dir, '{}-{}'.format(timestamp, exp_name))
os.makedirs(SAVE_DIR, exist_ok=True)

# Save args params into json file
with open(SAVE_DIR + '/parse_params.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Check if gpu training is available
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
   model = MLP_Net(input_dim, [1024, 1024, num_classes], batch_norm=nn.BatchNorm1d, use_scl=args.use_scl)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)

# Dataset specific
# malware_real: [10, 60, 90], gamma = 0.3
# malware_syn: noise-0.6-imb-20 [5, 30, 60], gamma = 0.3; other settings: [30, 60], gamma = 0.3
if args.dataset_origin == 'synthetic':
    milestones = [5, 30, 60] if args.noise_type == 'imb_step_0.05' and args.noise_rate == 0.6 else [30, 60]
else:
    milestones = [10, 60, 90]

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=milestones, gamma=0.3, last_epoch=-1) 
# useless
dist = [0.14, 0.15, 0.15, 0.12, 0.15, 0.09, 0.01, 0.12, 0.03, 0.02, 0.01, 0.01]

with torch.cuda.device(args.gpu_index):
    ourmatch = our_match(model=model, optimizer=optimizer, scheduler=lr_scheduler, \
                         logdir=SAVE_DIR, dist=dist, args=args)
    ourmatch.run(train_data, clean_targets, noisy_targets, \
                 train_loader, test_loader)