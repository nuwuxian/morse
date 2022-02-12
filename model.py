import torch.nn as nn
from copy import deepcopy
import torch
import torch.nn.functional as F

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
           module.bias.data.zero_()

class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

class MLP_Net(nn.Module):
    def __init__(self, input_dim, hiddens, batch_norm=None, use_scl=False):

        super(MLP_Net, self).__init__()

        self.encoder = nn.Sequential()
        self.classifier = nn.Sequential()
        self.use_scl = use_scl

        for i in range(len(hiddens)):
            bias = (batch_norm == None) or i == len(hiddens) - 1 
            if i == 0:
                self.encoder.add_module('mlp_%d' %i, nn.Linear(input_dim, hiddens[i], bias=bias))
            elif i != len(hiddens) - 1:
                self.encoder.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i], bias=bias))
            else:
                self.classifier.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i], bias=bias))

            if batch_norm is not None:
                if i != len(hiddens) - 1:
                    self.encoder.add_module('batchnorm_%d' % i, batch_norm(hiddens[i]))

            if i != len(hiddens) - 1:
                self.encoder.add_module('relu_%d' %i, nn.ReLU())
        if use_scl:
           self.head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
           )
        self.init()

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def forward_feat(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def init(self):
        self.encoder.apply(initialize_weights)
        self.classifier.apply(initialize_weights)
        if self.use_scl:
            self.head.apply(initialize_weights)
