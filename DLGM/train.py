
import torch
import torch.optim as optim
import torch.nn as nn
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
import numpy as np
import random
from data_gen import Datagen

from generator import Generator
from evaluation import evaluate_model, bin_plot
from recognition import Recognition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=None
print("Using device: ", device)

import torch
print(torch.__version__)

gen = Datagen(device)

x, y = gen.get_generated_data(seq_len=2)

print("x", x.size())
print("y", y)

import random

# Hyperparameters
sequence_length = [2*i for i in range(4,16)]  # 2-20 increments of two
hidden_layers = [1,2] # 1 and 2
hidden_1 = [2**i for i in range(2,7)] # 2^4 to 2^9
hidden_2 =[2**i for i in range(5,10)] # 2^2 to 2^5
variance = [0.001, 0.01, 0.005, 0.05]
lr = [0.001, 0.01, 0.1, 0.005] # stop at 0.005
data_probability = [i/5 for i in range(1,6)]
regularization = [1/i for i in range(1,10)]
for i in range(3):
    regularization.append(0)

epochs = 1000
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
                                options.append(entry)
                
                                         
random.shuffle(options)    



import torch.utils.data as data
from itertools import chain
import torch.nn.functional as F

def loss(y, y_hat, mean, R,g, device=None, seq_len=1):

    mse = nn.MSELoss().to(device)
    l = F.binary_cross_entropy(y_hat, y, reduction='sum')
    amount = mean[0].size()[0]*mean[0].size()[1]
    for m, r in zip(mean, R):

        C = r @ r.transpose(-2,-1) + 1e-6
        det = C.det() + 1e-6
        l += 0.5 * torch.sum(m.pow(2).sum(-1)
                             + C.diagonal(dim1=-2,dim2=-1).sum(-1)
                            -det.log()  -1)/amount

    #print(l, F.binary_cross_entropy(x_hat, x, reduction='sum'))
    return l

best_model = None
best_score = 10000000000000000
batch_size = 10
best_history= [0,0,0,0,0,0]
for entry in options:

    x_d, y_d = gen.get_generated_data(entry["seq_len"], entry["variance"], entry["data_prob"])
    x_t, y_t = gen.get_true_data(entry["seq_len"])
    x_val, y_val = gen.get_test_data(entry["seq_len"])



    model_g = Generator(hidden_size=entry["hidden"],
                        latent_dim=entry["latent"],
                        output_dim=y_d.size()[1],
                        layers=entry["layers"],
                        seq_len=entry["seq_len"],
                        device=device)
    model_r = Recognition(input_dim=x_d.size()[1],
                          latent_dim=entry["latent"],
                          layers=entry["layers"],
                          device=device)

    loader = data.DataLoader(data.TensorDataset(x_d, y_d), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(chain(model_r.parameters(), model_g.parameters()), lr=entry["l"])
    history = []
    bce = nn.BCELoss().to(device)
    for e in range(epochs):
        model_g.train()
        model_r.train()

        for x, y in loader:

            x.to(device)
            y.to(device)
            if x.size()[0] < batch_size:
                continue
            if random.random() < 0.5:
                continue

            rec = model_r(x)
            model_g.set_xi(rec[-1])
            y_hat = model_g()
            l = loss(y, y_hat, rec[0], rec[1], device, entry["seq_len"])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()



        if e % 10 != 0:
            continue

        count = 0
        sum_loss = [0, 0]
        for j in range(2):
            for x, y in loader:
                model_g.eval()
                model_r.eval()
                model_g.make_xi()
                with torch.no_grad():
                    rec = model_r(x)
                    model_g.set_xi(rec[-1])
                    b = model_g()
                    l = loss(y, b, rec[0], rec[1], device, entry["seq_len"])
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
            evaluate_model(best_model,model_r, x_t, y_t, x_val,y_val,  entry)
    else:
        with torch.no_grad():
            evaluate_model(model_g,model_r,x_t, y_t ,x_val,y_val, entry)
        print("Old model still stands:\nCurrent loss: ", history[-1], "\nBest loss:", best_history[-1])


