# -*- coding: utf-8 -*-
"""

MITBIH dataset loader

Author:
Date modified: 

"""


#necessory import libraries

import os 
import sys 
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#class names and corresponding labels of the MITBIH dataset
cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
            'Unknown':3, 'Fusion Beats':4}

reverse_cls_dit = {0:'Non-Ectopic Beats', 1:'Superventrical Ectopic', 2:'Ventricular Beats',
                   3: 'Unknown', 4:'Fusion Beats'}


class mitbih_oneClass(Dataset):
    """
    A pytorch dataloader loads on class data from mithib dataset.
    Example Usage:
        class0 = mitbih_oneClass(class_id = 0)
        class1 = mitbih_oneClass(class_id = 1)
    """
    def __init__(self, file_path='./mitbih_train.csv', reshape = True, class_id = 0):
        data_pd = pd.read_csv(file_path, header=None)
        data = data_pd[data_pd[187] == class_id]
    
        self.data = data.iloc[:, :128].values  # change it to only has 128 timesteps to match conv1d dim changes
        self.labels = data[187].values
        
        if reshape:
            self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1])
        
        print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    

class mitbih_allClass(Dataset):
    def __init__(self, file_path='./mitbih_train.csv', isBalanced = False, n_samples=20000, reshape = True):
        data_train = pd.read_csv(file_path, header=None)
        
        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0]
        data_1 = data_train[data_train[187] == 1]
        data_2 = data_train[data_train[187] == 2]
        data_3 = data_train[data_train[187] == 3]
        data_4 = data_train[data_train[187] == 4]
        
        if isBalanced:
            data_0_resample = resample(data_0, n_samples=n_samples, 
                               random_state=123, replace=True)
            data_1_resample = resample(data_1, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_2_resample = resample(data_2, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_3_resample = resample(data_3, n_samples=n_samples, 
                                       random_state=123, replace=True)
            data_4_resample = resample(data_4, n_samples=n_samples, 
                                       random_state=123, replace=True)

            train_dataset = pd.concat((data_0_resample, data_1_resample, 
                                      data_2_resample, data_3_resample, data_4_resample))
        else:
            train_dataset = pd.concat((data_0, data_1, 
                                      data_2, data_3, data_4))

        self.X_train = train_dataset.iloc[:, :128].values
        if reshape:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
            
        self.y_train = train_dataset[187].values
            
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        
        if isBalanced:
            print(f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, \
                  {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')
        else:
            print(f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    

import matplotlib.pyplot as plt

num_signals_to_sample = 3
def draw_graphs(signals, filename):

    dim = signals.shape[1]
    assert signals.shape[0] > num_signals_to_sample, f"Not enough signals in class {filename}"
        
    random_indexs = random.sample(range(len(signals)), num_signals_to_sample)
    sampled_signals = signals[random_indexs]
    
    # plot the sampled signals in a line
    fig, ax = plt.subplots(1, 3, figsize=(10,3))
    for i in range(num_signals_to_sample):
        for d in range(dim):
            ax[i].plot(sampled_signals[i][d][:])
    
    # save the plot with the same name as the numpy file
    fig.savefig(filename + '.pdf')
    plt.close(fig)


if __name__ == "__main__":
#     file_path = '/home/x_l30/Research/datasets/MITBIH/mitbih_test.csv'
#     mitbih_oneClass(file_path = file_path, class_id=0)
#     mitbih_oneClass(file_path = file_path, class_id=1)
#     mitbih_allClass(file_path = file_path)
    
    
#     file_path = '/home/x_l30/Research/datasets/MITBIH/mitbih_train.csv'
#     mitbih_oneClass(file_path = file_path, class_id=0)
#     mitbih_oneClass(file_path = file_path, class_id=1)
#     mitbih_allClass(file_path = file_path)
 

    file_path = '/home/x_l30/Research/datasets/MITBIH/mitbih_train.csv'    
    class_ids = [0, 1, 2, 3, 4]

    for class_id in class_ids:
        class_signals = mitbih_oneClass(file_path = file_path, class_id=class_id).data
        draw_graphs(class_signals, "MITBIH_real_class{}".format(class_id))




    