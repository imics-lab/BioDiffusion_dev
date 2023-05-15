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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############## load real dataset #############
# load UNIMIB dataset
data_path = '/home/x_l30/Research/datasets/UniMiB/UniMiB-SHAR/data'
unimib_Walking = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='Walking')
unimib_Jumping = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='Jumping')
unimib_SittingDown = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='SittingDown')

unimib_StandingUpFS = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='StandingUpFS')
unimib_StandingUpFL = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='StandingUpFL')
unimib_Running = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='Running')
unimib_GoingUpS = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='GoingUpS')
unimib_GoingDownS = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='GoingDownS')
unimib_LyingDownFS = datasets.unimib_dataLoader(path_in = data_path, single_class = True, class_name='LyingDownFS')



############# load UNIMIB diffusion model
def generate_synthetic_signals(checkpoint, num_samples = 100):
    model = modules.Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 3).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = modules.GaussianDiffusion1D(
        model,
        seq_length = 144,
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

############# generate synthetic UNIMIB data ############

#Walking
syn_Walking = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_Walking/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_Walking.npy'), 'wb') as f:
    np.save(f, syn_Walking)
visualization.visualization (unimib_Walking.one_class_train_data, syn_Walking, 'umap', os.path.join(saving_graph_path, 'unimib_Walking_umap'))
print("Generate Walking done!\n")

# Jumping
syn_Jumping = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_Jumping/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_Jumping.npy'), 'wb') as f:
    np.save(f, syn_Jumping)
visualization.visualization(unimib_Jumping.one_class_train_data, syn_Jumping, 'umap', os.path.join(saving_graph_path, 'unimib_Jumping_umap'))
print("Generate Jumping done!\n")

# SittingDown
syn_SittingDown = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_SittingDown/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_SittingDown.npy'), 'wb') as f:
    np.save(f, syn_SittingDown)
visualization.visualization(unimib_SittingDown.one_class_train_data, syn_SittingDown, 'umap', os.path.join(saving_graph_path, 'unimib_SittingDown_umap'))
print("Generate SittingDown done!\n")

# StandingUpFS
syn_StandingUpFS = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_StandingUpFS/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_StandingUpFS.npy'), 'wb') as f:
    np.save(f, syn_StandingUpFS)
visualization.visualization(unimib_StandingUpFS.one_class_train_data, syn_StandingUpFS, 'umap', os.path.join(saving_graph_path, 'unimib_StandingUpFS_umap'))
print("Generate StandingUpFS done!\n")

# StandingUpFL
syn_StandingUpFL = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_StandingUpFL/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_StandingUpFL.npy'), 'wb') as f:
    np.save(f, syn_StandingUpFL)
visualization.visualization(unimib_StandingUpFL.one_class_train_data, syn_StandingUpFL, 'umap', os.path.join(saving_graph_path, 'unimib_StandingUpFL_umap'))
print("Generate StandingUpFL done!\n")

# Running
syn_Running = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_Running/checkpoint.pt'), 2000)
with open(os.path.join(saving_data_path, 'syn_Running.npy'), 'wb') as f:
    np.save(f, syn_Running)
visualization.visualization(unimib_Running.one_class_train_data, syn_Running, 'umap', os.path.join(saving_graph_path, 'unimib_Running_umap'))
print("Generate Running done!\n")

# GoingUpS
syn_GoingUpS = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_GoingUpS/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_GoingUpS.npy'), 'wb') as f:
    np.save(f, syn_GoingUpS)
visualization.visualization(unimib_GoingUpS.one_class_train_data, syn_GoingUpS, 'umap', os.path.join(saving_graph_path, 'unimib_GoingUpS_umap'))
print("Generate GoingUpS done!\n")

# GoingDownS
syn_GoingDownS = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_GoingDownS/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_GoingDownS.npy'), 'wb') as f:
    np.save(f, syn_GoingDownS)
visualization.visualization(unimib_GoingDownS.one_class_train_data, syn_GoingDownS, 'umap', os.path.join(saving_graph_path, 'unimib_GoingDownS_umap'))
print("Generate GoingDownS done!\n")

# LyingDownFS
syn_LyingDownFS = generate_synthetic_signals(os.path.join(current_path, './checkpoint/DDPM1D_Uncondtional_UNIMIB_LyingDownFS/checkpoint.pt'), 1500)
with open(os.path.join(saving_data_path, 'syn_LyingDownFS.npy'), 'wb') as f:
    np.save(f, syn_LyingDownFS)
visualization.visualization(unimib_LyingDownFS.one_class_train_data, syn_LyingDownFS, 'umap', os.path.join(saving_graph_path, 'unimib_LyingDownFS_umap'))
print("Generate LyingDownFS done!\n")
