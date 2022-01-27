import torch.nn as nn
import numpy as np
from copy import deepcopy

import torch


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
    def __init__(self, input_dim, hiddens, batch_norm=None):

        super(MLP_Net, self).__init__()

        self.encoder = nn.Sequential()
        self.classfier = nn.Sequential()


        for i in range(len(hiddens)):
            bias = (batch_norm == None) or i == len(hiddens) - 1 
            if i == 0:
                self.encoder.add_module('mlp_%d' %i, nn.Linear(input_dim, hiddens[i], bias=bias))
            elif i != len(hiddens) - 1:
                self.encoder.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i], bias=bias))
            else:
                self.classfier.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i], bias=bias))

            if batch_norm is not None:
                if i != len(hiddens) - 1:
                    self.encoder.add_module('batchnorm_%d' % i, batch_norm(hiddens[i]))

            if i != len(hiddens) - 1:
                self.encoder.add_module('relu_%d' %i, nn.ReLU())

    def forward(self, x):
        return self.classifier(self.encoder(x))


    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_classifier(self, x):
        return self.classifier(x)