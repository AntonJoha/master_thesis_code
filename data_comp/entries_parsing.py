import json
import matplotlib.pyplot as plt
import numpy as np
def get_file(filename="entries"):
    l = None
    with open(filename, "r") as f:
        l = json.loads(f.read())
    return l


def get_labels(struct):
    return list(struct.keys())

def get_values_for_label(struct, label="", unique=True):
    to_return = []
    for s in struct:
        if unique and s[label] in to_return:
            continue
        to_return.append(s[label])
    to_return.sort()
    return to_return

def get_entries_with_value(struct, label="", value=""):
    to_return = []
    for s in struct:
        if str(s[label]) == str(value):
            to_return.append(s)
    return to_return


def _count_entries(l):
    to_return = {}
    for i in l:
        if i not in to_return:
            to_return[i] = 0
        to_return[i] += 1
    keys = list(to_return.keys())
    keys.sort()
    new = {}
    # Sorting the dictionary
    for k in keys:
        new[k] = to_return[k]
    return new

def get_yvalue_based_on_label(struct, x_label, y_label):
    to_return = {}
    for s in struct:
        curr = s[x_label]
        if curr not in to_return:
            to_return[curr] = []
        to_return[curr].append(s[y_label])
    keys = list(to_return.keys())
    keys.sort()
    new = {}
    # Sorting the dictionary
    for k in keys:
        new[k] = to_return[k]
    return new



def box_plots(struct, x_label, y_label, title, x_text=None, y_text=None,folder="plots/", trim_labels=False):
    if x_text is None:
        x_text = x_label
    if y_text is None:
        y_text = y_label

    x_values = get_values_for_label(struct, x_label)
    values = get_yvalue_based_on_label(struct, x_label, y_label)
    data = []
    for i in values:
        data.append(values[i])
    fig, ax = plt.subplots()
    print(len(x_values), len(data))
    if trim_labels:
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])[0:4]
    ax.boxplot(data, labels=x_values)
    ax.set_ylabel(y_text)
    ax.set_xlabel(x_text)
    ax.set_title(title)
    filename = folder + str(x_text) + "_" + str(y_text) + "_" +  title + "_" + "boxplots.png"
    fig.savefig(filename)


def multiple_box_plots(struct_1, struct_2, x_label, y_label, label_1, label_2, title, x_text=None, y_text=None, folder="plots/", trim_labels=False):
    
    if x_text is None:
        x_text = x_label
    if y_text is None:
        y_text = y_label
    
    color1 = "lightblue"
    color2 = "lightgreen"
    
    fig, ax = plt.subplots()
    
    x_values = get_values_for_label(struct_1, x_label)
    x = np.arange(0, len(x_values))
    values = get_yvalue_based_on_label(struct_1, x_label, y_label)
    data = []
    for i in values:
        data.append(values[i])
        
    if trim_labels:
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])[0:4]
    b1 = ax.boxplot(data, labels=x_values,widths=0.1,positions=x-0.1, patch_artist=True)
    
    for c in b1["boxes"]:
        c.set_facecolor(color1)
        
        
        
    x_values = get_values_for_label(struct_2, x_label)
    values = get_yvalue_based_on_label(struct_2, x_label, y_label)
    data = []
    for i in values:
        data.append(values[i])
        
    if trim_labels:
        for i in range(len(x_values)):
            x_values[i] = str(x_values[i])[0:4]
    b2 = ax.boxplot(data, labels=[""]*len(x_values),widths=0.1,positions=x+0.1, patch_artist=True)
    
    for c in b2["boxes"]:
        c.set_facecolor(color2)
                       
    
    ax.legend([b1["boxes"][0], b2["boxes"][0]], [label_1, label_2])
    
    ax.set_ylabel(y_text)
    ax.set_xlabel(x_text)
    ax.set_title(title)
    filename = folder + str(x_text) + "_" + str(y_text) + "_" + str(label_1) + str(label_2) +  title + "_" + "boxplots.png"
    fig.savefig(filename)
    
    
    
def boxplot_two_values(data, labels, x_text, y_text, title, color=["lightgreen", "lightblue"], color_v=["lightgreen", "lightblue"],folder="plots/",outliers=False):
    
    fig, ax = plt.subplots()
    
    b = ax.boxplot(data, labels=labels,  patch_artist=True, showfliers=outliers)
    
    for box, c in zip(b["boxes"], color):
        
        box.set_facecolor(c)
    
    ax.set_xlabel(x_text)
    ax.set_ylabel(y_text)
    ax.set_title(title)
    
    filename = folder + str(x_text) + "_" + str(y_text) + "_" + str(title) + "outliers" if outliers else "" +  "multipleboxplots.png"
    fig.savefig(filename)
    
    
    
    fig, ax = plt.subplots()
    
    p = ax.violinplot(data)
    for body, c in zip(p["bodies"],color_v):
        body.set_facecolor(c)
        
    ax.set_xticks([y + 1 for y in range(len(labels))], labels=labels)
    
    