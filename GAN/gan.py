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

    def __init__(self, input_dim=5, output_dim=1, layers=2):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.make_layers()
        self.lstm_zero()
        self.sig = nn.Sigmoid()


    def make_layers(self):
        operations = []
        self.L = nn.Linear(in_features=2, out_features=self.input_dim).double()
        self.T = nn.ModuleList()
        self.G = nn.ModuleList()
        for i in range(self.layers):
            self.T.append(nn.LSTM(input_size=self.input_dim, hidden_size=self.input_dim).double())
            self.G.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim).double())
            
        self.last = nn.Linear(in_features=self.input_dim, out_features=self.output_dim).double()

    def forward(self, z):
        #print(z.size()[0])
        z = z.double()
        z = self.L(z)
        count = 0
        for t, g in zip(self.T, self.G):
            res, self.internal_state[count] = t(z, self.internal_state[count])
            mid_val = g(torch.randn(z.size()[0], self.input_dim).double())
            z = res + mid_val

        return self.sig(self.last(z))

    def lstm_zero(self):
        self.internal_state = []
        #first and last is not an lstm layer.
        for layer in self.T:
            self.internal_state.append([torch.zeros(1, self.input_dim).double(),
                                        torch.zeros(1, self.input_dim).double()])

    def init_state(self):
        for i,e in enumerate(self.internal_state):
            h = e[0].detach()
            c = e[1].detach()
            self.internal_state[i] = (h, c)

