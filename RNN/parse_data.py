#/usr/bin/env python3
import pandas as pd
import torch
import numpy as np


data_path = "../data/openb_pod_list_default.csv"


def get_data():
    df = pd.read_csv(data_path)
    
    df.drop(labels=["pod_phase",
                    "num_gpu",
                    "cpu_milli",
                    "gpu_spec",
                    "qos",
                    "deletion_time",
                    "memory_mib",
                    "scheduled_time",
                    "name",
                    ],
            axis=1,inplace=True)

    return df


def make_data(df):
    
    x_train, y_train = [], []
    prev = -1
    
    m = df.max()[0]
    print("Max value: ", m)
    count = 0
    for row in df.values:
        x_train.append([count])
        count += 1
        y_train.append([row[0]/m])
    return torch.tensor(x_train, dtype=torch.double)/count,torch.tensor(y_train, dtype=torch.double)


def find_closest(val, options):
    index = 0
    min = 10000000
    for i in range(len(options)):
        if np.abs(val-options[i][0]) < min:
            min = np.abs(val-options[i][0])
            index = i
    return index

def get_modified_values(df):

    dic = {}

    for row in df.values:
        if row[0] not in dic:
            dic[row[0]] = 1
        else:
            dic[row[0]] += 1

    dic_sort = sorted(dic.items(), key=lambda x:x[1])
    dic_decreasing = list(reversed(dic_sort))

    values = []
    for i in range(10):
        values.append(dic_decreasing[i])

    val = []

    for row in df.values:
        closest = find_closest(row[0], values)
        val.append([1.0 if i == closest else 0.0 for i in range(len(values))])

    return torch.tensor(val)

