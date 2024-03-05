import matplotlib.pyplot as plt
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
from datetime import datetime
import torch
import pickle
import numpy as np
import json
import mauve 


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
    fig.savefig("images_old/%s%s.png" % ("forcing" if forcing else "not_forcing", str(conf)))


def get_lots_yhat(m,x):
    res = []
    m.eval()
    m.clean_state()
    prev = x[0]
    for i in range(100000):
        un = prev.unsqueeze(0)
        val = m(un)
        prev = torch.cat([prev[1:], val[0]], dim=0)
        res.append(val.detach().cpu()[0])
        
    return res

def plot_lots(m, x, conf):
    y = get_lots_yhat(m,x)
    timesteps = 100000
    entries = 200
    opt = [int(i*timesteps/5) for i in range(1, 5)]
    fig, ax = plt.subplots(len(opt), figsize=(12,12))
    
    for i in range(len(opt)):
        ax[i].plot(range(opt[i], opt[i] + entries),y[opt[i]:(opt[i]+entries)])
    
    fig.savefig("images_old/100k%s.png" % conf)
    
    
    

def get_yhat(m,x, forcing=True):
    res = []
    m.eval()
    m.random_state()
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

def bin_plot(y, low, high, conf, step_size):
    count = {}
    values = np.around(np.arange(low, high + step_size*2, step_size), decimals=2)
    for i in values:
        count[i] = 0
    for i in y:
        prev = values[0]
        for j in values:
            if i < j:
                count[prev] += 1
                break
            prev = j
            
    names = list(count.keys())
    values = list(count.values())
    
    fig, ax = plt.subplots(figsize=(12,12))
    ax.bar(range(len(count) -1), values[:-1], tick_label=names[:-1])
    fig.savefig("images_old/%s%s.png" % ("barfig_non_forcing", str(conf)))

   
def add_run(to_add, filename="entries"):
    
    entries = None
    
    try:
        with open(filename, "r") as f:
            entries = json.loads(f.read())
    except:
        # If there is no file
        entries = []
    entries.append(to_add)
    
    with open(filename, "w") as f:
        f.write(json.dumps(entries, indent=4))
    
           
def evaluate_model(m,x,y,conf, draw_images=True):
    conf_str = str(conf).replace(" ", "")
    plt.close('all')
    total_sum = 5717.8652
    
    conf_str += datetime.now().strftime("%Y%M%D%H%M%S").replace("/","").replace(":","")

    y_hat_f = get_yhat(m,x, forcing=True)    
    y_hat = get_yhat(m,x,forcing=False)

    to_add = conf
    #to_add["y_hat_f"] = y_hat_f.cpu().numpy().tolist()
    #to_add["y_hat"] = y_hat.cpu().numpy().tolist()

   
    if draw_images:
        draw_model(y_hat_f, y, conf_str,forcing=True)
        draw_model(y_hat, y, conf_str, forcing=False)
        bin_plot(y_hat_f, 0, 1, conf_str, 0.05)
        plot_lots(m, x, conf_str)

