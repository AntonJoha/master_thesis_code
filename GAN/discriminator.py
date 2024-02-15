import numpy as np


from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



class Discriminator(nn.Module):

    def dummy_function(self, dont=None, care=None):
        return torch.zeros(self.input_dim)

    def __init__(self, input_dim=1, output_dim=1, hidden_size=100):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size).double()
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_dim).double()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()


    def forward(self, z):
        #print(z.size()[0])
        y, self.internal_state = self.lstm(z, self.internal_state)
        
        # We only care about predicting with the last value. 
        y = self.relu(y)
        y = self.linear(y)
        return self.sig(y)

    def random_state(self):
        self.internal_state = [torch.randn(1, 100).double(),
                               torch.randn(1, 100).double()]



    def init_state(self):
        h = self.internal_state[0].detach()
        c = self.internal_sself.internal_state.detach()
        self.internal_state = (h, c)

