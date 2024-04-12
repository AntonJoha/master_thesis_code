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


def draw_model( y_hat ,y, conf, forcing=True, extra_str=""):

    fig, ax = plt.subplots(2)
    
    ax[0].plot(range(1,51), y_hat[:50])
    ax[0].plot( range(1,51) , y[:50].cpu())

    ax[1].plot(range(2000,2050), y_hat[2000:2050])
    
    ax[1].plot(range(2000,2050), y[2000:2050].cpu() )
    fig.suptitle("%s" % conf)
    fig.savefig("images/%s%s%s.png" % ( str(conf), "forcing" if forcing else "not_forcing", extra_str))

def draw_test_model( y_hat ,y, conf, forcing=True, extra_str=""):

    fig, ax = plt.subplots(2)
    
    ax[0].plot(range(1,51), y_hat[:50])
    ax[0].plot( range(1,51) , y[:50].cpu())

    ax[1].plot(range(450,500), y_hat[450:])
    
    ax[1].plot(range(450,500), y[450:].cpu() )
    fig.suptitle("%s" % conf)
    fig.savefig("images/%s%s%s.png" % ( str(conf), "forcing" if forcing else "not_forcing", extra_str))

    

def get_lots_yhat(m,x):
    res = []
    m.eval()
    m.clean_state()
    prev = x[0]
    for i in range(100000):
        un = prev.unsqueeze(0)
        val = m(un)
        prev = torch.cat([prev[1:], val], dim=0)
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
    
    fig.savefig("images/%s_100k.png" % conf)
    
    
    

def get_yhat(m,x, forcing=True):
    res = []
    m.eval()
    prev = x[0]
    for i in x:
        m.clean_state()

        un = prev.unsqueeze(0)
        val = m(un)
        if forcing:
            prev = i
        else:
            prev = torch.cat([prev[1:], val], dim=0)
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

def get_bin(y, low, high, conf, step_size):
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
            
    
    return count

def bin_plot(y, low, high, conf, step_size, extra_str=""):
    count = get_bin(y, low, high, conf, step_size)
    names = list(count.keys())
    values = list(count.values())
    fig, ax = plt.subplots(figsize=(12,12))
    ax.bar(range(len(count) -1), values[:-1], tick_label=names[:-1])
    fig.savefig("images/%s%s.png" % (str(conf), "barfig_" + extra_str ))

   
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
    

def mkdir(path):
    import os
    try:
        os.system("mkdir %s" % ("images/" + path))
    except:
        print("Folder already exist")
        
def evaluate_model(m,x,y,x_test,y_test,conf, draw_images=True):
    conf_str = str(conf).replace(" ", "").replace("/","").replace(":","").replace("'", "")
    model_name = conf_str
    plt.close('all')
    total_sum = 5717.8652
    
    mkdir(conf_str)
    conf_str += "/" + datetime.now().strftime("%Y%M%D%H%M%S").replace("/","").replace(":","")

    y_hat_f = get_yhat(m,x, forcing=True)    
    y_hat = get_yhat(m,x,forcing=False)
    y_hat_sum = sum(y_hat)
    y_hat_f_sum = sum(y_hat_f)
    
    y_hat_test_f = get_yhat(m,x_test, forcing=True)
    y_hat_test = get_yhat(m,x_test, forcing=False)
    
    to_add = conf
    to_add["y_hat_f"] = y_hat_f.cpu().numpy().tolist()
    to_add["y_hat"] = y_hat.cpu().numpy().tolist()
    
    y_hat_sums = [y_hat_sum.cpu().numpy().tolist()]
    y_hat_f_sums = [y_hat_f_sum.cpu().numpy().tolist()]
    y_hat_test_f_sums = [sum(y_hat_test_f).cpu().numpy().tolist()]
    y_hat_test_sums = [sum(y_hat_test).cpu().numpy().tolist()]
    
    for i in range(10):
        y_hat_f_t = get_yhat(m,x, forcing=True)    
        y_hat_t = get_yhat(m,x,forcing=False)
        y_hat_sums.append(sum(y_hat_t).cpu().numpy().tolist())
        y_hat_f_sums.append(sum(y_hat_f_t).cpu().numpy().tolist())
        y_hat_test_f_t = get_yhat(m,x_test, forcing=True)
        y_hat_test_t = get_yhat(m,x_test, forcing=False)
        y_hat_test_sums.append(sum(y_hat_test_t).cpu().numpy().tolist())
        y_hat_test_f_sums.append(sum(y_hat_test_f_t).cpu().numpy().tolist())
    
    print(y_hat_test_f_sums)

    to_add["y_hat_sums"] = y_hat_sums
    to_add["y_hat_f_sums"] = y_hat_f_sums
    to_add["y_hat_sum_var"] = np.var(y_hat_sums)
    to_add["y_hat_f_sum_var"] = np.var(y_hat_f_sums)
    to_add["y_hat_sum_mean"] = np.mean(y_hat_sums)
    to_add["y_hat_f_sum_mean"] = np.mean(y_hat_f_sums)
    to_add["y_hat_test_f_sum_mean"] = np.mean(y_hat_test_f_sums)
    to_add["y_hat_test_f_sum_var"] = np.var(y_hat_test_f_sums)
    to_add["y_hat_test_sum_mean"] = np.mean(y_hat_test_sums)
    to_add["y_hat_test_sum_var"] = np.var(y_hat_test_sums)
    to_add["y_hat_f_bar"] = get_bin(y_hat_f, 0, 1, conf_str, 0.05)
    to_add["y_hat_bar"] = get_bin(y_hat, 0, 1, conf_str, 0.05)

    
    a = (np.round(y.cpu().numpy() - 0.0001,decimals=4))
    a_str = ""
    for i in a:
        a_str += str(i[0]) + ","
    b = np.round(y_hat.cpu().numpy(), decimals=4)
    b_str = ""
    for i in b:
        b_str += str(i) + ","
   
    out = mauve.compute_mauve(p_text=a_str[:1000], q_text=b_str[-3000:], device_id=0, max_text_length=256, verbose=False)
    
    
    to_add["mauve"] = out.mauve
    
    b = np.round(y_hat_f.cpu().numpy(), decimals=4)
    b_str = ""
    for i in b:
        b_str += str(i) + ","
    
    out = mauve.compute_mauve(p_text=a_str[:1000], q_text=b_str[-3000:], device_id=0, max_text_length=256, verbose=False)
    
    
    to_add["mauve_f"] = out.mauve
    
    add_run(to_add)

    #save_model(m, y_hat,  model_name, y_hat_sum, y_hat_f_sum)
    
    if draw_images:
        draw_model(y_hat_f, y, conf_str,forcing=True)
        draw_model(y_hat, y, conf_str, forcing=False)
        draw_test_model(y_hat_test_f, y_test, conf_str,forcing=True, extra_str="test")
        draw_test_model(y_hat_test, y_test, conf_str,forcing=False,extra_str="test")
        bin_plot(y_hat_f, 0, 1, conf_str, 0.05, "forcing")
        bin_plot(y_hat, 0, 1, conf_str, 0.05, "non-forcing")

        #plot_lots(m, x, conf_str)

