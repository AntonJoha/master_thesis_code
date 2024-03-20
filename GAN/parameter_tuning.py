import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import random as random
import torch
import mauve 



def train_model(model_g,
                model_d,
                x_data,
                y_data,
                batch_size,
                seq_len,
                epochs,
                lr=0.001,
                loss=nn.MSELoss(),
                optimizer=optim.Adam,
                device=None,
                strict_teacher_forcing=True,
                random_state=True):
    
    
    true = torch.ones(batch_size, seq_len, 1).to(device)
    false = torch.zeros(batch_size, seq_len, 1).to(device)
    
    optimizer_g = optimizer(model_g.parameters(), lr=lr)
    optimizer_d = optimizer(model_d.parameters(), lr=lr)
    loader = data.DataLoader(data.TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=True)

    history = []
    bce = nn.BCELoss().to(device)

    
    for e in range(epochs):
        model_g.train()
        model_d.train()

   
        for x, y in loader:
            x.to(device)
            y.to(device)
            if x.size()[0] < batch_size:
                continue
            if random.random() < 0.5:
                continue
            model_g.lstm_zero(seq_len)
            y_pred = model_g(x).detach()
            l = bce(model_d(y_pred), false) + bce(model_d(x), true)
            optimizer_d.zero_grad()
            l.backward()
            optimizer_d.step()

          
           
        for x, y in loader:
            if x.size()[0] < batch_size:
                continue
            if random.random() < 0.5:
                continue
            model_g.lstm_zero(seq_len)
            y_pred = model_g(x)
            d = model_d(y_pred)
            
            l = bce(d, true)
            optimizer_g.zero_grad()
            l.backward()
            optimizer_g.step()

            
        
        
        if e % 10 != 0:
            continue
        
        count = 0
        sum_loss = [0, 0]
        for j in range(2):
            for x, y in loader:
                model_g.eval()
                with torch.no_grad():
                    y_pred = model_g(x)
                    res = []
                    for i in y_pred:
                        res.append([i[-1]])
                    #print(res)
                    s = torch.tensor(res).to(device)
                    #print(s)
                    #p = torch.cat(s)
                    #print(p)
                    l = loss(s, y).cpu()
                    sum_loss[j] += l.item()
                    count += 1
                    
        
        
        sum_loss[0] /= count
     
        
        history.append([e, sum_loss[0], sum_loss[1]])
        print(history[-1])

        if len(history) > 15:
            #if no real improvements are being done stop the training. 
            # but keep doing the training if the results without correctly feeding values get better
            if abs(history[-15][1] - history[-1][1]) < 0.0001:
                return model_g, history

    return model_g, history

