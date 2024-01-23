#/usr/bin/env python3
import pandas as pd


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




