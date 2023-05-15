# -*- coding: utf-8 -*-
"""

simulated dataset loader

Author:
Date modified: 

"""
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

class simulated_data(Dataset):
    def __init__(self, file_folder = './', class_id=0, all_class=False, train=True, reshape = True, is_normalize=True):
        if train:
            multiclass_data = np.load(os.path.join(file_folder, 'multiclass_X_train.npy'))
            multiclass_labels = np.load(os.path.join(file_folder, 'multiclass_y_train.npy'))
        else:
            multiclass_data = np.load(os.path.join(file_folder, 'multiclass_X_test.npy'))
            multiclass_labels = np.load(os.path.join(file_folder, 'multiclass_y_test.npy'))
            
        if reshape:
            multiclass_data = multiclass_data.reshape(multiclass_data.shape[0], 1, multiclass_data.shape[1])
            
        if is_normalize:
            multiclass_data = self.normalization(multiclass_data)
        
        if not all_class:
            self.signals = multiclass_data[multiclass_labels == class_id]
            self.labels = multiclass_labels[multiclass_labels == class_id]
            print(f'The simulated data shpe is {self.signals.shape}, class id is {class_id}\n',)
        else:
            self.signals = multiclass_data
            self.labels = multiclass_labels
            print(f'The simulated data shpe is {self.signals.shape}\n \
                    It has {len(self.labels[self.labels == 0])} class 0\n \
                    It has {len(self.labels[self.labels == 1])} class 1\n \
                    It has {len(self.labels[self.labels == 2])} class 2\n \
                    It has {len(self.labels[self.labels == 3])} class 3\n \
                    It has {len(self.labels[self.labels == 4])} class 4\n')
            
        
            
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
    
    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,:] = self._normalize(epochs[i,j,:])
                epochs[i,j,:] = self._min_max_normalize(epochs[i,j,:])

        return epochs
 



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
    file_folder = '/home/x_l30/Research/datasets/simulated/'
    simulated_data(file_folder = file_folder, class_id = 0)
    simulated_data(file_folder, class_id = 1)
    simulated_data(file_folder, all_class=True)

    
    
    file_folder = '/home/x_l30/Research/datasets/simulated/'  
    class_ids = [0, 1, 2, 3, 4]

    for class_id in class_ids:
        class_signals = simulated_data(file_folder = file_folder, class_id=class_id).signals
        draw_graphs(class_signals, "Simulated_real_class{}".format(class_id))
        
        
        
