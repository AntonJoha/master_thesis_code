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
        self.teacher_forcing = True

        self.lstm1 = nn.LSTMCell(input_size=input_size, hidden_size=self.hidden_size).to(device)
        self.lstm2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size).to(device)

        self.l1 = nn.Linear(self.hidden_size,self.hidden_size).to(device)
        self.l2 = nn.Linear(self.hidden_size,self.output_size).to(device)

        self.clean_state()

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    
    def forward_step(self, x):
        self.hidden1 = self.lstm1(x, self.hidden1)
        x = self.hidden1[0]
        self.hidden2 = self.lstm2(x, self.hidden2)
        x = self.hidden2[0]
        #x = self.relu(x)
        
        #x = self.l1(x)
        #x = self.relu(x)
        x = self.l2(x)
        return self.sig(x)
    
    def forward(self, x):
        res = []
        n = x[0]
        for i in x:
            if self.teacher_forcing:
                res.append(self.forward_step(i))
            else:
                res.append(self.forward_step(n))
                n = res[-1]
                
        return torch.stack(res)
    
    def init_state(self):
        self.h_0 = self.hidden1[0].detach()
        self.c_0 = self.hidden1[1].detach()
        self.hidden1 = (self.h_0, self.c_0)
        self.h_0 = self.hidden2[0].detach()
        self.c_0 = self.hidden2[1].detach()
        self.hidden2 = (self.h_0, self.c_0)
    
    
    def clean_state(self):
        self.h_0 = torch.zeros(self.hidden_size, device=self.device)
        self.c_0 = torch.zeros(self.hidden_size, device=self.device)
        self.hidden1 = (self.h_0, self.c_0)
        self.h_0 = torch.zeros(self.hidden_size, device=self.device)
        self.c_0 = torch.zeros(self.hidden_size, device=self.device)
        self.hidden2 = (self.h_0, self.c_0)


    def random_state(self):
        self.h_0 = torch.randn(self.hidden_size, device=self.device)
        self.c_0 = torch.randn(self.hidden_size, device=self.device)
        self.hidden1 = (self.h_0, self.c_0)
        self.h_0 = torch.randn(self.hidden_size, device=self.device)
        self.c_0 = torch.randn(self.hidden_size, device=self.device)
        self.hidden2 = (self.h_0, self.c_0)
        

