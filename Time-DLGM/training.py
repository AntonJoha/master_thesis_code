
import torch
import torch.optim as optim
import torch.nn as nn
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
import numpy as np
import random
from data_gen import Datagen
from recognition import Recognition
from generator import Generator
from evaluation import evaluate_model, bin_plot
from time_recognition import TimeRecognition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=None
print("Using device: ", device)

import torch
print(torch.__version__)
 
gen = Datagen(device)

x, y, x_1 = gen.get_generated_data(seq_len=2)

print("x", x[0])
print("y", y[0])
print("x_1", x_1[0])

import random

# Hyperparameters
sequence_length = [2*i for i in range(4,16)] # 2-20 increments of two
hidden_layers = [1,2] # 1 and 2
hidden_1 = [2**i for i in range(2,4)] # 2^4 to 2^9
hidden_2 =[2**i for i in range(5,10)] # 2^2 to 2^5
variance = [0.001, 0.01, 0.005, 0.05]
lr = [0.001, 0.01, 0.1, 0.005] # stop at 0.005
data_probability = [i/5 for i in range(1,6)]
regularization = [1/i for i in range(1,10)]
for i in range(3):
    regularization.append(0)

epochs = 500
optimizer = [optim.Adam, optim.SGD]

options = []

for seq_len in sequence_length:
    for layers in hidden_layers:
        for h1 in hidden_1:
            for h2 in hidden_2:
                for l in lr:
                    for v in variance:
                        for p in data_probability:
                            for r in regularization:
                                entry = {}
                                entry["seq_len"] = seq_len
                                entry["layers"] = layers
                                entry["latent"] = h1
                                entry["hidden"] = h2
                                entry["l"] = l
                                entry["variance"] = v
                                entry["data_prob"] = p
                                entry["regularization"] = r
                                options.append(entry)
                
                                         
random.shuffle(options)    


import torch.utils.data as data
from itertools import chain
import torch.nn.functional as F

def loss(x, x_hat, mean, R, s, x_1,reg,  device=None, seq_len=1):
    
    mse = nn.MSELoss().to(device)
    l = F.binary_cross_entropy(x_hat, x, reduction='sum')
    amount = mean[0].size()[0]*mean[0].size()[1]
    for m, r in zip(mean, R):
        
        C = r @ r.transpose(-2,-1) + 1e-6
        det = C.det() + 1e-6 
        l += 0.5 * torch.sum(m.pow(2).sum(-1) 
                             + C.diagonal(dim1=-2,dim2=-1).sum(-1)
                            -det.log()  -1)/amount

    count = len(s)*2
    for a, b in zip(s, x_1):
        l += reg*mse(a[0], b[0])/count
        l += reg*mse(a[1], b[1])/count
    
    #print(l, F.binary_cross_entropy(x_hat, x, reduction='sum'))
    return l 

best_model = None
best_score = 10000000000000000
batch_size = 10
best_history= [0,0,0,0,0,0]
for entry in options:

    x_d, y_d, x_1_d = gen.get_generated_data(entry["seq_len"], entry["variance"], entry["data_prob"])
    x_t, y_t, x_t_1 = gen.get_true_data(entry["seq_len"])
    x_val, y_val, x_val_1 = gen.get_test_data(entry["seq_len"])


    model_t = TimeRecognition(input_dim=x_d[0].size()[1],
                              hidden_size=entry["hidden"],
                              seq_len=entry["seq_len"],
                              layers=entry["layers"],
                             device=device)

    model_g = Generator(hidden_size=entry["hidden"],
                        latent_dim=entry["latent"],
                        output_dim=y_d[0].size()[0],
                        layers=entry["layers"],
                        seq_len=entry["seq_len"],
                        device=device)
    model_r = Recognition(input_dim=x_d[0].size()[1],
                          latent_dim=entry["latent"],
                          layers=entry["layers"],
                          device=device)

    loader = data.DataLoader(data.TensorDataset(x_d, y_d, x_1_d), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(chain(model_r.parameters(), model_g.parameters(), model_t.parameters()), lr=entry["l"])
    #optimizer = optim.Adam(model_r.parameters())
    history = []
    bce = nn.BCELoss().to(device)
    for e in range(epochs):
        model_g.train()
        model_r.train()
        model_t.train()


        for x, y, x_1 in loader:

            x.to(device)
            y.to(device)
            if x.size()[0] < batch_size:
                continue
            if random.random() < 0.5:
                continue

            t = model_t(x)
            t_1 = model_t(x_1)
            model_g.make_internal_state()
            rec = model_r(x_1)
            model_g.set_xi(rec[-1])
            model_g.set_internal_state(t)
            b, s = model_g()

            l = loss(y, b, rec[0], rec[1], s, t_1, entry["regularization"], device, entry["seq_len"])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
                        

        
        if e % 10 != 0:
            continue
        
        count = 0
        sum_loss = [0, 0]
        for j in range(2):
            for x, y, x_1 in loader:
                model_g.eval()
                model_t.eval()
                model_r.eval()
                model_g.make_internal_state()
                model_g.make_xi()
                with torch.no_grad():
                    model_g.make_internal_state()
                    rec = model_r(x_1)
                    t = model_t(x)
                    t_1 = model_t(x_1)
                    model_g.set_internal_state(t)
                    model_g.set_xi(rec[-1])
                    b,s = model_g()
                    l = loss(y, b, rec[0], rec[1],s,t_1,entry["regularization"], device, entry["seq_len"])
                    res = []
                    
                    sum_loss[j] += l.item()
                    count += 1
                    
        
        
        sum_loss[0] /= count
     
        
        history.append([e, sum_loss[0], sum_loss[1]])
        print(history[-1])

        if len(history) > 15:
            #if no real improvements are being done stop the training. 
            # but keep doing the training if the results without correctly feeding values get better
            if abs(history[-15][1] - history[-1][1]) < 0.0001:
                break
        
    
    if history[-1][1] < best_score:
        print("New best model:\nNew loss: ", history[-1], "\nOld loss:", best_history[-1], "\nHistory:" , history[-10:])
        best_model = model_g
        best_history = history
        best_score = history[-1][1]
        best_config = entry
        with torch.no_grad():
            evaluate_model(best_model,model_r, model_t, x_t, y_t, x_t_1,x_val,y_val, x_val_1, entry)
    else:
        with torch.no_grad():
            evaluate_model(model_g,model_r, model_t, x_t, y_t, x_t_1,x_val,y_val, x_val_1, entry)
        print("Old model still stands:\nCurrent loss: ", history[-1], "\nBest loss:", best_history[-1])
    
