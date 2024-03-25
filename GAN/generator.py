from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



class Generator(nn.Module):

    def dummy_function(self, dont=None, care=None):
        return torch.zeros(self.input_dim)

    def __init__(self, input_dim=50, output_dim=1, layers=10, g_size=50, device=None):
        super(Generator, self).__init__()

        self.device=device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g_size = g_size
        self.layers = layers
        self.make_layers()
        self.lstm_zero()
        self.sig = nn.Sigmoid()


    def get_g(self):
        
        to_return = nn.Sequential(
                nn.Linear(in_features=self.g_size, out_features=self.g_size).to(self.device),
                nn.ReLU(),
                nn.Linear(in_features=self.g_size, out_features=self.g_size).to(self.device),
                )
        return to_return

    def make_layers(self):
        operations = []
        self.L = nn.Linear(in_features=self.input_dim, out_features=self.g_size).to(self.device)
        self.T = nn.ModuleList()
        self.G = nn.ModuleList()
        for i in range(self.layers):
            self.T.append(nn.LSTM(input_size=self.g_size, hidden_size=self.g_size, batch_first=True).to(self.device))
            self.G.append(self.get_g())
            
        self.last = nn.Linear(in_features=self.g_size, out_features=self.output_dim).to(self.device)

    def forward(self, z):
        z = z
        z = self.L(z)
        count = 0
        for t, g in zip(self.T, self.G):
            res, self.internal_state[count] = t(z, self.internal_state[count])
            mid_val = g(torch.randn(z.size()[0], res.size()[1], res.size()[2]).to(self.device))
            z = res + mid_val
        
        z = self.last(z)
        return self.sig(z)

    def lstm_zero(self, batch_size=1):
        self.internal_state = []
        #first and last is not an lstm layer.
        for layer in self.T:
            self.internal_state.append([torch.zeros(1,batch_size, self.g_size).to(self.device),
                                        torch.zeros(1, batch_size, self.g_size).to(self.device)])

    def init_state(self, batch_size=1):
        for i,e in enumerate(self.internal_state):
            h = e[0].detach()
            c = e[1].detach()
            self.internal_state[i] = (h, c)

