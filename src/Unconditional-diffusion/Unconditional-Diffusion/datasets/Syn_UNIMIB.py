# synthetic UNIMIB dataset dataloader

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}
reverse_class_dict = {0:'StandingUpFS',1:'StandingUpFL', 2:'Walking', 3:'Running', 4:'GoingUpS', 5:'Jumping', 6:'GoingDownS', 7:'LyingDownFS', 8:'SittingDown'}

class_names = ['StandingUpFS','StandingUpFL','Walking','Running','GoingUpS','Jumping','GoingDownS','LyingDownFS','SittingDown']
real_sample_size = [119,169,1394,1572,737,600,1068,228,168]
max_sample_size = [1500, 1500, 1500, 2000, 1500, 1500, 1500, 1500, 1500]


class Synthetic_UNIMIB(Dataset):
    def __init__(self, folder_path, classes, sample_size):
        self.folder_path = folder_path
        self.classes = classes
        self.sample_size = sample_size
        self.data, self.labels = self.get_data()
        
        print(f'data shape is {self.data.shape} labels shape is {self.labels.shape}')
        
    
    def get_data(self):
        data = []
        labels = []
        for idx, cls in enumerate(self.classes):
            npy_file = os.path.join(self.folder_path, f'syn_{cls}.npy')
            class_data = np.load(npy_file)[:self.sample_size[idx]]
            class_labels = np.full((len(class_data), ), idx)
            data.append(class_data)
            labels.append(class_labels)
            print(f'Sampled {len(class_data)} in {cls} class\n')
            
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return data, labels

    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
            


if __name__ == "__main__":
    dataset1 = Synthetic_UNIMIB(folder_path = '/home/x_l30/Research/Unconditional-Diffusion/synthetic/synthetic_data/', classes = class_names, sample_size = real_sample_size)
    
    dataset2 = Synthetic_UNIMIB(folder_path = '/home/x_l30/Research/Unconditional-Diffusion/synthetic/synthetic_data/', classes = class_names, sample_size = max_sample_size)
    
    dataloader = DataLoader(dataset2, batch_size=4, shuffle=True)
    for i, (signal, label) in enumerate(dataloader):
        print(signal.shape)
        print(label)
        
        if i > 3:
            break