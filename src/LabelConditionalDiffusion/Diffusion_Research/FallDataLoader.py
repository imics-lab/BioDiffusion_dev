import numpy as np
import pandas as pd
import os
from scipy import signal
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


columns_to_select = ['x-axis (g)', 'y-axis (g)', 'z-axis (g)']
# Function to process a single DataFrame and separate signals
def process_data(data):
    fall_signals = []
    not_fall_signals = []

    current_signal = []
    current_label = None

    for index, row in data.iterrows():
        if current_label is None:
            current_label = row['outcome']

        if row['outcome'] != current_label:
            if current_label == 1:
                fall_signals.append(np.array(current_signal))
            else:
                not_fall_signals.append(np.array(current_signal))

            current_signal = []
            current_label = row['outcome']

        current_signal.append(row[columns_to_select])

    if current_label == 1:
        fall_signals.append(np.array(current_signal))
    else:
        not_fall_signals.append(np.array(current_signal))

    return fall_signals, not_fall_signals



# Function to resample multi-channel signals
def resample_multichannel_signal(sig, target_length):
    sig = sig.reshape(sig.shape[1], sig.shape[0])
    num_channels = sig.shape[0]
    resampled_signal = np.zeros((num_channels, target_length))
    
    for i in range(num_channels):
        resampled_signal[i, :] = signal.resample(sig[i, :], target_length)
        
    return resampled_signal


# assume you have multiple signals stored in a 3D NumPy array called `signals`
# with shape (num_signals, num_samples, num_channels)
def standarize_signals(signals):
    # Step 1: standardize each signal to a 0-1 scale
    min_values = np.min(signals, axis=(0, 1)) # get the minimum value of each channel
    max_values = np.max(signals, axis=(0, 1)) # get the maximum value of each channel
    standardized_signals = (signals - min_values) / (max_values - min_values)

    # Step 2: store the minimum and maximum values of each channel
    # you'll need these later to convert the signals back to their original scale
    min_values = np.tile(min_values, (signals.shape[0], signals.shape[1], 1))
    max_values = np.tile(max_values, (signals.shape[0], signals.shape[1], 1))

    # Step 3: perform any necessary calculations on the standardized signals
    # ...

    # Step 4: convert the standardized signals back to their original scale
    # original_scale_signals = standardized_signals * (max_values - min_values) + min_values

    return standardized_signals, min_values, max_values


class Fall_NotFall_loader(Dataset):
    def __init__(self, folder_path = '/home/x_l30/Diffusion/Diffusion_Research/datasets/FallData', seq_lenth=96):
        all_fall_signals = []
        all_not_fall_signals = []

        # Iterate through all CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_file = os.path.join(folder_path, filename)
                data = pd.read_csv(csv_file)

                # Assuming the data point column is named 'data_point' and the label column is named 'label'
                # Change the column names accordingly if they are different in your CSV files

                fall_signals, not_fall_signals = process_data(data)
                all_fall_signals.extend(fall_signals)
                all_not_fall_signals.extend(not_fall_signals)
        
        #remove not fall data that has length = 1
        filtered_not_fall_signals = np.array([signal for signal in all_not_fall_signals if len(signal) != 1])
        
        # Resample the signals to the target length
        resampled_fall_signals = [resample_multichannel_signal(sig, seq_lenth) for sig in all_fall_signals]
        resampled_not_fall_signals = [resample_multichannel_signal(sig, seq_lenth) for sig in filtered_not_fall_signals]
        
        # change to numpy array
        resampled_fall_signals = np.array(resampled_fall_signals)
        resampled_not_fall_signals = np.array(resampled_not_fall_signals)
        
        self.standardized_fall_signals, self.fall_min_values, self.fall_max_values = standarize_signals(resampled_fall_signals)
        self.standardized_not_fall_signals, self.not_fall_min_values, self.not_fall_max_values = standarize_signals(resampled_not_fall_signals)
        
        self.fall_labels = np.ones(len(self.standardized_fall_signals))
        self.not_fall_labels = np.zeros(len(self.standardized_not_fall_signals))
        
        self.all_data = np.concatenate((self.standardized_fall_signals, self.standardized_not_fall_signals), axis=0)
        self.all_labels = np.concatenate((self.fall_labels, self.not_fall_labels))
        
        print(self.all_data.shape)
        print(self.all_labels.shape)
        
    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]
    