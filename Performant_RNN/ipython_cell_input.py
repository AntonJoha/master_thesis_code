import numpy as np
import torch.optim as optim
import torch.utils.data as data
from IPython.display import clear_output
import torch.nn as nn
import random

batch_size = 100
x_d, y_d = make_data(get_binary_values(get_data()), device)


print(y_d.size(), x_d.size())

model = PredictTime(input_size=x_d[0].size()[0],
                    output_size=y_d[0].size()[0],
                    hidden_layers=1,
                    hidden_size=25, device=device).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.1)

loss = nn.BCELoss()
loader = data.DataLoader(data.TensorDataset(x_d,y_d), batch_size=batch_size)
epochs = 100
for e in range(epochs):
    model.train()
    #print(next(iter(loader)))
    model.clean_state()
    res = []
    
    
    if random.random() < e/(epochs*2):
        model.teacher_forcing = False
    else:
        model.teacher_forcing = True
    
    for x, y in loader:
        
        if random.random() < 0.5:
            continue
        model.random_state()
        
        
        y_pred = model(x)
        l = loss(y_pred, y)
        res.append(l)
        #print(y_pred, y)
    
    
    l = res[0]
    for i in res[1:]:
        l += i
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    if e % 10 != 0:
        continue
    #clear_output(wait=True)
    sum_loss = 0
    #print(list(model.parameters())[-1])

    for x, y in loader:
        model.eval()
        with torch.no_grad():
            y_pred = model(x)
            sum_loss += np.sqrt(loss(y_pred, y).cpu())
    
    print("Epoch %d Loss %.4f" % (e, sum_loss))
    
    
#for d in df.values:
