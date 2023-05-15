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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############## load real dataset #############
# load UNIMIB dataset
data_path =  '/home/x_l30/Research/datasets/simulated/'
simu_cls0 = datasets.simulated_data(file_folder = data_path, class_id = 0)
simu_cls1 = datasets.simulated_data(file_folder = data_path, class_id = 1)
simu_cls2 = datasets.simulated_data(file_folder = data_path, class_id = 2)
simu_cls3 = datasets.simulated_data(file_folder = data_path, class_id = 3)
simu_cls4 = datasets.simulated_data(file_folder = data_path, class_id = 4)

############# load simulated diffusion model
def generate_synthetic_signals(checkpoint, num_samples = 100):
    model = modules.Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8, 8),
    channels = 1).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = modules.GaussianDiffusion1D(
        model,
        seq_length = 512,
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

############# generate synthetic simulated data ############

# class 0
syn_simu_class0 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_simulated_class0/checkpoint.pt'), 5000)
with open(os.path.join(saving_data_path, 'syn_simu_class0.npy'), 'wb') as f:
    np.save(f, syn_simu_class0)
visualization.visualization (simu_cls0.signals, syn_simu_class0, 'umap', os.path.join(saving_graph_path, 'unimib_simu_class0_umap'))
print("Generate syn_simu_class0 done!\n")


# Class 1
syn_simu_class1 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_simulated_class1/checkpoint.pt'), 5000)
with open(os.path.join(saving_data_path, 'syn_simu_class1.npy'), 'wb') as f:
    np.save(f, syn_simu_class1)
visualization.visualization(simu_cls1.signals, syn_simu_class1, 'umap', os.path.join(saving_graph_path, 'unimib_simu_class1_umap'))
print("Generate syn_simu_class1 done!\n")

# Class 2
syn_simu_class2 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_simulated_class2/checkpoint.pt'), 5000)
with open(os.path.join(saving_data_path, 'syn_simu_class2.npy'), 'wb') as f:
    np.save(f, syn_simu_class2)
visualization.visualization(simu_cls2.signals, syn_simu_class2, 'umap', os.path.join(saving_graph_path, 'unimib_simu_class2_umap'))
print("Generate syn_simu_class2 done!\n")

# Class 3
syn_simu_class3 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_simulated_class3/checkpoint.pt'), 5000)
with open(os.path.join(saving_data_path, 'syn_simu_class3.npy'), 'wb') as f:
    np.save(f, syn_simu_class3)
visualization.visualization(simu_cls3.signals, syn_simu_class3, 'umap', os.path.join(saving_graph_path, 'unimib_simu_class3_umap'))
print("Generate syn_simu_class3 done!\n")

# Class 4
syn_simu_class4 = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_simulated_class4/checkpoint.pt'), 5000)
with open(os.path.join(saving_data_path, 'syn_simu_class4.npy'), 'wb') as f:
    np.save(f, syn_simu_class4)
visualization.visualization(simu_cls4.signals, syn_simu_class4, 'umap', os.path.join(saving_graph_path, 'unimib_simu_class4_umap'))
print("Generate syn_simu_class4 done!\n")

