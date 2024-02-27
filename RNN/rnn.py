import torch.nn as nn
import torch

class PredictTime(nn.Module):
    def __init__(self, input_size=1,output_size=1,hidden_layers=1,hidden_size=1, device=None):

        super().__init__()
        

        self.device=device

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size1, num_layers=1).to(device) # two lstm different hidden size
        if hidden_layers == 2:
            self.lstm2 = nn.LSTM(input_size=self.hidden_size1, hidden_size=self.hidden_size2, num_layers=1).to(device)
        self.l = nn.Linear(self.hidden_size,self.output_size).to(device)

        self.clean_state()

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU() #GELU
        
    def forward(self, x):
 
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.l2(x)
        return self.sig(x)
    
    def init_state(self):
        self.h_0 = self.hidden[0].detach()
        self.c_0 = self.hidden[1].detach()
        self.hidden = (self.h_0, self.c_0)
    
    def clean_state(self):
        self.h_0 = torch.zeros(self.hidden_layers, self.hidden_size, device=self.device)
        self.c_0 = torch.zeros(self.hidden_layers,  self.hidden_size, device=self.device)
        self.hidden = (self.h_0, self.c_0)


    def random_state(self):
        self.h_0 = torch.randn(self.hidden_layers,self.hidden_size, device=self.device)
        self.c_0 = torch.randn(self.hidden_layers,self.hidden_size, device=self.device)
        self.hidden = (self.h_0, self.c_0)

