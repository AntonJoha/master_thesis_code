import numpy as np


from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



class Discriminator(nn.Module):

    def dummy_function(self, dont=None, care=None):
        return torch.zeros(self.input_dim)

    def __init__(self, input_dim=1, output_dim=1, hidden_size=100, layers=1, device=None):
        super(Discriminator, self).__init__()
        

        self.device=device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_dim)
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()


    def forward(self, z):        
        
        y, self.internal_state = self.lstm(z)
        
        # We only care about predicting with the last value. 
        y = self.relu(y)
        y = self.linear(y)
        return self.sig(y)

    def random_state(self, batch_size=1):
        self.internal_state = [torch.randn(1,batch_size, self.hidden_size).to(self.device),
                               torch.randn(1,batch_size, self.hidden_size).to(self.device)]



    def init_state(self):
        h = self.internal_state[0].detach()
        c = self.internal_sself.internal_state.detach()
        self.internal_state = (h, c)

