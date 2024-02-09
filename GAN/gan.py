import argparse
import os
import numpy as np
import math


from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



class Generator(nn.Module):

    def dummy_function(self, dont=None, care=None):
        return torch.zeros(self.input_dim)

    def __init__(self, input_dim=5, output_dim=1, layers=10):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.operations = self.make_layers()
        self.lstm_zero()
        self.sig = nn.Sigmoid()


    def make_layers(self):
        operations = []
        self.L = nn.Linear(in_features=2, out_features=self.input_dim).double()

        for i in range(self.layers):
            l = [ nn.LSTM(input_size=self.input_dim, hidden_size=self.input_dim).double(),
                               nn.Linear(in_features=self.input_dim, out_features=self.output_dim).double()]
            l = nn.ModuleList(l)
            operations.append(l)
        self.last = nn.Linear(in_features=self.input_dim, out_features=self.output_dim).double()
        return  nn.ModuleList(operations)

    def forward(self, z):
        z = z.double()
        z = self.L(z)

        for i, e in enumerate(self.operations):
            res, self.internal_state[i] = e[0](z, self.internal_state[i])
            mid_val = e[1](torch.normal(mean=torch.tensor([0 for i in range(self.input_dim)]).double(),
                                        std=torch.tensor([1 for i in range(self.input_dim)]).double()
                                        )
                           )
            z = torch.add(res, mid_val)
        return self.sig(self.last(z))

    def lstm_zero(self):
        self.internal_state = []
        #first and last is not an lstm layer.
        print(self.operations[0][0].all_weights[0][0][0][0])
        for layer in self.operations:
            self.internal_state.append([torch.zeros(1, self.input_dim).double(),
                                        torch.zeros(1, self.input_dim).double()])

    def init_state(self):
        for i,e in enumerate(self.internal_state):
            h = e[0].detach()
            c = e[1].detach()
            self.internal_state[i] = (h, c)

