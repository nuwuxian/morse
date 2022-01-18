import torch.nn as nn
import numpy as np

class MLP_Net(nn.Module):
    def __init__(self, input_dim, hiddens, batch_norm=None):

        super(MLP_Net, self).__init__()
        self.mlp = nn.Sequential()

        for i in range(len(hiddens)):
            bias = (batch_norm == None) or i == len(hiddens) - 1 
            if i == 0:
                self.mlp.add_module('mlp_%d' %i, nn.Linear(input_dim, hiddens[i], bias=bias))
            else:
                self.mlp.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i], bias=bias))

            if batch_norm is not None:
                if i != len(hiddens) - 1:
                    self.mlp.add_module('batchnorm_%d' % i, batch_norm(hiddens[i]))

            if i != len(hiddens) - 1:
                self.mlp.add_module('relu_%d' %i, nn.ReLU())

    def forward(self, x):
        return self.mlp(x)