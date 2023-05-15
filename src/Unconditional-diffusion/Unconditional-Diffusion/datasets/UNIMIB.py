# -*- coding: utf-8 -*-
"""

UniMiB_SHAR_ADL dataset loader

Author:
Date modified: 

"""

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}
reverse_class_dict = {0:'StandingUpFS',1:'StandingUpFL', 2:'Walking', 3:'Running', 4:'GoingUpS', 5:'Jumping', 6:'GoingDownS', 7:'LyingDownFS', 8:'SittingDown'}

class unimib_dataLoader(Dataset):
    def __init__(self,
        path_in = './UniMiB-SHAR/data',
        incl_xyz_accel = True, #include component accel_x/y/z in ____X data
        incl_rms_accel = False, #add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                               #False => combine test & validation groups
        is_normalize = True,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        one_hot_encode = False, data_mode = 'Train', single_class = False, class_name= 'Walking', augment_times = None, seq_length = 144):
        
        self.incl_xyz_accel = incl_xyz_accel
        self.incl_rms_accel = incl_rms_accel
        self.incl_val_group = incl_val_group
        self.split_subj = split_subj
        self.one_hot_encode = one_hot_encode
        self.data_mode = data_mode
        self.class_name = class_name
        self.single_class = single_class
        self.is_normalize = is_normalize
        
        
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

        #Reshape data and compute total (rms) acceleration
        num_samples = 151 
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        if (self.incl_rms_accel):
            rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
            adl_data = np.dstack((adl_data,rms_accel))
        #remove component accel if needed
        if (not self.incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
            
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html


        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels

        if (not self.incl_val_group):
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj'] + 
                                            self.split_subj['validation_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]
        else:
            train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]

            validation_index = np.nonzero(np.isin(sub_num, self.split_subj['validation_subj']))
            x_validation = adl_data[validation_index]
            y_validation = act_num[validation_index]

        test_index = np.nonzero(np.isin(sub_num, self.split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]

        
        #If selected one-hot encode y_* using keras to_categorical, reference:
        #https://keras.io/api/utils/python_utils/#to_categorical-function and
        #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        if (self.one_hot_encode):
            y_train = self.to_categorical(y_train, num_classes=9)
            if (self.incl_val_group):
                y_validation = self.to_categorical(y_validation, num_classes=9)
            y_test = self.to_categorical(y_test, num_classes=9)

            
        # reshape x_train, x_test data shape from (BH, length, channel) to (BH, channel, length)
        self.x_train = np.transpose(x_train, (0, 2, 1))
        self.x_train = self.x_train[:,:,:144]
        self.y_train = y_train
        
        self.x_test = np.transpose(x_test, (0, 2, 1))
        self.x_test = self.x_test[:,:,:144]
        self.y_test = y_test
        
        if not self.single_class:
            print(f'x_train shape is {self.x_train.shape}, x_test shape is {self.x_test.shape}')
            print(f'y_train shape is {self.y_train.shape}, y_test shape is {self.y_test.shape}')
            print(f'StandingUpFS label is 0 has {len(y_train[y_train == 0])} train smaples, {len(y_test[y_test == 0])} test samples, \n \
                    StandingUpFL label is 1 has {len(y_train[y_train == 1])} train smaples, {len(y_test[y_test == 1])} test samples, \n \
                    Walking label is 2 has {len(y_train[y_train == 2])} train smaples, {len(y_test[y_test == 2])} test samples, \n \
                    Running label is 3 has {len(y_train[y_train == 3])} train smaples, {len(y_test[y_test == 3])} test samples, \n \
                    GoingUS label is 4 has {len(y_train[y_train == 4])} train smaples, {len(y_test[y_test == 4])} test samples, \n \
                    Jumping label is 5 has {len(y_train[y_train == 5])} train smaples, {len(y_test[y_test == 5])} test samples, \n \
                    GoingDownS label is 6 has {len(y_train[y_train == 6])} train smaples, {len(y_test[y_test == 6])} test samples, \n \
                    LyingDownFS label is 7 has {len(y_train[y_train == 7])} train smaples, {len(y_test[y_test == 7])} test samples, \n \
                    SittingDown label is 8 has {len(y_train[y_train == 8])} train smaples, {len(y_test[y_test == 8])} test samples \n')
        
        
        if self.is_normalize:
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)
        
        #Return the give class train/test data & labels
        if self.single_class:
            self.one_class_train_data = self.x_train[self.y_train == class_dict[self.class_name]]
            self.one_class_train_labels = self.y_train[self.y_train == class_dict[self.class_name]]
            self.one_class_test_data = self.x_test[self.y_test == class_dict[self.class_name]]
            self.one_class_test_labels = self.y_test[self.y_test == class_dict[self.class_name]]
            
            print(f'return single class data and labels, class is {self.class_name}')
            print(f'train_data shape is {self.one_class_train_data.shape}, test_data shape is {self.one_class_test_data.shape}')
            print(f'train label shape is {self.one_class_train_labels.shape}, test data shape is {self.one_class_test_labels.shape}')

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]       

    
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
    
    def __len__(self):
        
        if self.data_mode == 'Train':
            if self.single_class:
                return len(self.one_class_train_labels)
            else:
                return len(self.y_train)
        else:
            if self.single_class:
                return len(self.one_class_test_labels)
            else:
                return len(self.y_test)
        
    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            if self.single_class:
                return self.one_class_train_data[idx], self.one_class_train_labels[idx]
            else:
                return self.x_train[idx], self.y_train[idx]
        else:
            if self.single_class:
                return self.one_class_test_data[idx], self.one_class_test_labels[idx]
            else:
                return self.x_test[idx], self.y_test[idx]
            
    
import matplotlib.pyplot as plt
import random

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

    import matplotlib.pyplot as plt
    
    data_path = '/home/x_l30/Research/datasets/UniMiB/UniMiB-SHAR/data'
    # unimib = unimib_dataLoader(path_in = data_path, single_class = True, class_name= 'Walking')
    # unimib = unimib_dataLoader(path_in = data_path, single_class = True, class_name= 'Running')
    # unimib = unimib_dataLoader(path_in = data_path, single_class = False)
    
    unimib = unimib_dataLoader(path_in = data_path, single_class = True, class_name= 'Walking')
    
    
    class_names = ['StandingUpFS','StandingUpFL','Walking','Running','GoingUpS','Jumping','GoingDownS','LyingDownFS','SittingDown']

    for class_name in class_names:
        class_signals = unimib_dataLoader(path_in = data_path, single_class = True, class_name= class_name).x_train
        draw_graphs(class_signals, "UNIMIB_real_{}".format(class_name))
    
    
    
