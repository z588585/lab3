import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from argparse import ArgumentParser

import numpy as np

def get_deterministic_seed(file_name, base_seed=42):
    seed = base_seed
    for char in file_name:
        seed = (seed * 31 + ord(char)) % 10000
    return seed


def list_files(folder_path):
    all_files = os.listdir(folder_path)
    sub_folders = [f for f in all_files if os.path.isdir(
        os.path.join(folder_path, f))]
    sub_files = [f for f in all_files if f.endswith('.csv')]
    return sub_folders, sub_files

def split_to_x_y(data):#not used
    try:
        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]
    except Exception as e:
        print('if you use wrong dataset, you will get this error:', e)
        sys.exit(1)
    return X, Y

def split_data_to_train_test(data, train_data_frac=0.7, split_random_seed=888): #not used
    train_data = data.sample(frac=train_data_frac,
                             random_state=split_random_seed)
    test_data = data.drop(train_data.index)
    return train_data, test_data


