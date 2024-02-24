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

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.hidden_layers).to(device)
        self.l1 = nn.Linear(self.hidden_size,self.hidden_size).to(device)
        self.l2 = nn.Linear(self.hidden_size,self.output_size).to(device)

        self.clean_state()

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
 
        x, self.hidden = self.lstm(x, self.hidden)
        #x = self.relu(x)
        
        #x = self.l1(x)
        #x = self.relu(x)
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

