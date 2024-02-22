import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import random as random
import torch


def train_model(model,
                x_data,
                y_data,
                sequence_length,
                epochs,
                lr=0.001,
                loss=nn.MSELoss(),
                optimizer=optim.Adam,
                device=None,
                strict_teacher_forcing=True,
                random_state=True):
    
    batch_size = sequence_length
    opt = optimizer(model.parameters(), lr=lr)
    loader = data.DataLoader(data.TensorDataset(x_data, y_data), batch_size=batch_size)

    history = []

    for e in range(epochs):
        model.train()

        model.clean_state()

        # We might want to train with purely teacher forcing, or with a combination. 
        # This allows for both.
        if strict_teacher_forcing == False:
            if random.random() < e/(epochs*2):
                model.teacher_forcing = False
            else:
                model.teacher_forcing = True

        for x, y in loader:

            # Either a random state is picked, or the previous
            # The idea behind a random state is that the model should be more tolerable and able to correct from "odd" values 
            if random_state:
                model.random_state()
            else:
                model.init_state()

            y_pred = model(x)
            l = loss(y_pred, y)
            opt.zero_grad()
            l.backward()
            opt.step()

        if e % 5 != 0:
            continue
        
        sum_loss = [0, 0]
        for i in range(2):
            for x, y in loader:
                model.eval()
                with torch.no_grad():
                    y_pred = model(x)
                    sum_loss[i] += loss(y_pred, y).cpu()
            model.teacher_forcing = False
    
        history.append([e, sum_loss[0], sum_loss[1]])

        if len(history) > 5:
            # if no real improvements are being done stop the training. 
            if abs(history[-5][1] - history[-1][1]) < 0.2 and abs(history[-5][2] - history[-1][2]) < 0.2:
                return model, history

    return model, history

