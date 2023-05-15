# -*- coding: utf-8 -*-
"""

python script used to generate synsthetic data

Author:
Date modified: 

"""

import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import visualization
import modules
import torch
import os


############## load devices #################
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############## load real dataset #############
# load MITBIH dataset
data_path = '/home/x_l30/Research/datasets/MITBIH/mitbih_train.csv'
mitbih_cls0 = datasets.mitbih_oneClass(file_path = data_path, class_id=0)
mitbih_cls1 = datasets.mitbih_oneClass(file_path = data_path, class_id=1)
mitbih_cls2 = datasets.mitbih_oneClass(file_path = data_path, class_id=2)
mitbih_cls3 = datasets.mitbih_oneClass(file_path = data_path, class_id=3)
mitbih_cls4 = datasets.mitbih_oneClass(file_path = data_path, class_id=4)



############# load MITBIH diffusion model
def generate_synthetic_signals(checkpoint, num_samples = 100):
    model = modules.Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = modules.GaussianDiffusion1D(
        model,
        seq_length = 128,
        timesteps = 1000,
        objective = 'pred_v').to(device)
    
    ckp = torch.load(checkpoint)
    model.load_state_dict(ckp['model_state_dict'])
    
    sampled_signals = diffusion.sample(batch_size = num_samples)
    print(sampled_signals.shape)
    sampled_signals = sampled_signals.to('cpu').detach().numpy()
    
    return sampled_signals

current_path = os.getcwd()
print(f"Current path: {current_path}")

saving_graph_path = './synthetic/visualizations/'
saving_data_path = './synthetic/synthetic_data/'

############# generate synthetic MITBIH data ############

#Class 0 'Non-Ectopic Beats'
syn_class0 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_MITBIH_class0/checkpoint.pt'), 7000)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class0.npy'), 'wb') as f:
    np.save(f, syn_class0)
visualization.visualization (mitbih_cls0.data[:1000], syn_class0, 'umap', os.path.join(saving_graph_path, 'MITBIH_class0_umap'))
print("Generate class 0 done!\n")

# Class 1 'Superventrical Ectopic'
syn_class1 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_MITBIH_class1/checkpoint.pt'), 7000)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class1.npy'), 'wb') as f:
    np.save(f, syn_class1)
visualization.visualization(mitbih_cls1.data[:1000], syn_class1, 'umap', os.path.join(saving_graph_path, 'MITBIH_class1_umap'))
print("Generate class 1 done!\n")


# Class 2 'Ventricular Beats'
syn_class2 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_MITBIH_class2/checkpoint.pt'), 7000)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class2.npy'), 'wb') as f:
    np.save(f, syn_class2)
visualization.visualization(mitbih_cls2.data[:1000], syn_class2, 'umap', os.path.join(saving_graph_path, 'MITBIH_class2_umap'))
print("Generate class 2 done!\n")

# Class 3 'Unknown'
syn_class3 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_MITBIH_class3/checkpoint.pt'), 7000)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class3.npy'), 'wb') as f:
    np.save(f, syn_class3)
visualization.visualization(mitbih_cls3.data, syn_class3, 'umap', os.path.join(saving_graph_path, 'MITBIH_class3_umap'))
print("Generate class 3 done!\n")

# Class 4 'Fusion Beats'
syn_class4 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_MITBIH_class4/checkpoint.pt'), 7000)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class4.npy'), 'wb') as f:
    np.save(f, syn_class4)
visualization.visualization(mitbih_cls4.data[:1000], syn_class4, 'umap', os.path.join(saving_graph_path, 'MITBIH_class4_umap'))
print("Generate class 4 done!\n")
