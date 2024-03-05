import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import random as random
import torch
import mauve 



def train_model(model,
                x_data,
                y_data,
                batch_size,
                epochs,
                lr=0.001,
                loss=nn.MSELoss(),
                optimizer=optim.Adam,
                device=None,
                strict_teacher_forcing=True,
                random_state=True):
    
    opt = optimizer(model.parameters(), lr=lr)
    loader = data.DataLoader(data.TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=True)

    history = []

    for e in range(epochs):
        model.train()

        model.clean_state()

        # We might want to train with purely teacher forcing, or with a combination. 
        # This allows for both.
           
        for x, y in loader:
            
            if random.random() < 0.5:
                continue
        
            model.random_state()
            y_pred = model(x)
            l = loss(y_pred, y)
            opt.zero_grad()
            l.backward()
            opt.step()

        if e % 10 != 0:
            continue
        
        count = 0
        sum_loss = [0, 0]
        for i in range(2):
            for x, y in loader:
                model.eval()
                with torch.no_grad():
                    y_pred = model(x)
                    sum_loss[i] += loss(y_pred, y).cpu()
                    count += 1
                    
            model.teacher_forcing = False
        
        res = []
        model.eval()
        model.clean_state()
        prev = x[0]
        for i in x:
            un = prev.unsqueeze(0)

        
            val = model(un)
            prev = torch.cat([prev[1:], val[0]], dim=0)
            res.append(val.detach()[0])
        
        res = torch.tensor(res,device=device).to(device)
        
        sum_loss[1] = loss(res, y.cpu())
        
        #Need to account for the fact that different sequence lenghts are used
        sum_loss[0] /= count
        
        #ptext = str([i[0] for i in y.cpu().numpy().tolist()])
        #qtext = str([i.cpu().numpy().tolist() for i in res])

        #out = mauve.compute_mauve(p_text=ptext, q_text=qtext, device_id=0, max_text_length=256, verbose=False)
       
        #print(out.mauve)

        
        history.append([e, sum_loss[0], sum_loss[1]])
        print(history[-1])

        if len(history) > 10:
            #if no real improvements are being done stop the training. 
            # but keep doing the training if the results without correctly feeding values get better
            if abs(history[-10][1] - history[-1][1]) < 0.000002 and history[-1][2] - history[-10][2] < 0.0002:
                return model, history

    return model, history

