import matplotlib.pyplot as plt
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
from datetime import datetime
import torch
import pickle

class Result():
    

    def __init__(self,model, yhat, config):
        self.yhat = yhat
        self.config = config
        self.model = model

def draw_model( y_hat ,y, conf, forcing=True):

    fig, ax = plt.subplots(2)
    
    ax[0].plot(range(1,51), y_hat[:50])
    ax[0].plot( range(1,51) , y[:50].cpu())

    ax[1].plot(range(2000,2050), y_hat[2000:2050])
    
    ax[1].plot(range(2000,2050), y[2000:2050].cpu() )
    fig.suptitle("%s" % conf)
    timestamp = datetime.now().strftime("%Y%M%D%H%M%S").replace("/","").replace(":","")
    fig.savefig("images/%s%s.png" % ("forcing" if forcing else "not_forcing", str(conf) + timestamp))


    
def get_yhat(m,x, forcing=True):
    res = []
    m.eval()
    m.clean_state()
    prev = x[0]
    for i in x:
        un = prev.unsqueeze(0)
        val = m(un)
        if forcing:
            prev = i
        else:
            prev = torch.cat([prev[1:], val[0]], dim=0)
        res.append(val.detach().cpu()[0])
    return torch.tensor(res)
    
def save_model(m, y_hat, conf, y_hat_sum, y_hat_f_sum):
    r = Result(m, y_hat, conf)
    r.y_hat_sum = y_hat_sum
    r.y_hat_f_sum = y_hat_f_sum
    
    timestamp = datetime.now().strftime("%Y%M%D%H%M%S").replace("/","").replace(":","")
    with open("models/%s%s.pickle" % (timestamp, conf), "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    d = None
    with open("models/%s%s.pickle" % (timestamp, conf), "rb") as f:
        d = pickle.load(f)
    for a,b in zip(r.yhat, d.yhat):
        if a != b:
            print("ERROR SAVING MODEL, model is not the same ")
            exit()
            
def evaluate_model(m,x,y,conf):
    
    total_sum = 5717.8652
    
    y_hat_f = get_yhat(m,x, forcing=True)
    y_hat = get_yhat(m,x,forcing=False)
    draw_model(y_hat_f, y, conf,forcing=True)
    draw_model(y_hat, y, conf, forcing=False)
    
    y_hat_sum = sum(y_hat)
    y_hat_f_sum = sum(y_hat_f)
    print(y_hat_sum)
    print(y_hat_f_sum)
    
    save_model(m, y_hat, conf, y_hat_sum, y_hat_f_sum)
    