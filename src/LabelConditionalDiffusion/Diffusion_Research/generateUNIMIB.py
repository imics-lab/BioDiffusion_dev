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
from UNIMIB import *
import visualization
import modules
import torch
import os
from modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free

############## load devices #################
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



############## load real dataset #############
# load UNIMIB dataset
unimib_Walking = unimib_load_dataset(single_class = True, class_name='Walking')
unimib_Jumping = unimib_load_dataset( single_class = True, class_name='Jumping')
unimib_SittingDown = unimib_load_dataset( single_class = True, class_name='SittingDown')

unimib_StandingUpFS = unimib_load_dataset( single_class = True, class_name='StandingUpFS')
unimib_StandingUpFL = unimib_load_dataset( single_class = True, class_name='StandingUpFL')
unimib_Running = unimib_load_dataset( single_class = True, class_name='Running')
unimib_GoingUpS = unimib_load_dataset( single_class = True, class_name='GoingUpS')
unimib_GoingDownS = unimib_load_dataset( single_class = True, class_name='GoingDownS')
unimib_LyingDownFS = unimib_load_dataset( single_class = True, class_name='LyingDownFS')

class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}

############# load UNIMIB diffusion model
def generate_synthetic_signals(checkpoint, num_samples = 100, class_name = 'Walking'):
    model = Unet1D_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = 9,
        cond_drop_prob = 0.5,
        channels = 3).to(device)

    
    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length = 144,
        timesteps = 1000).to(device)
    
    ckp = torch.load(checkpoint)
    model.load_state_dict(ckp['model_state_dict'])
    
    y = torch.Tensor([class_dict[class_name]] * num_samples).long().to(device)
    x = diffusion.sample(classes = y, cond_scale = 3.)

    sampled_signals = x.to('cpu').detach().numpy()
    print(sampled_signals.shape)
    
    return sampled_signals

current_path = os.getcwd()
print(f"Current path: {current_path}")

saving_graph_path = './synthetic/visualizations/'
saving_data_path = './synthetic/synthetic_data/'

############# generate synthetic UNIMIB data ############
ckpt = './checkpoint/DDPM1D_cls_free_UNIMIB/checkpoint.pt'

# #Walking
# syn_Walking = generate_synthetic_signals(ckpt, 1500, class_name = 'Walking')
# with open(os.path.join(saving_data_path, 'syn_Walking.npy'), 'wb') as f:
#     np.save(f, syn_Walking)
# visualization.visualization (unimib_Walking.one_class_train_data, syn_Walking, 'umap', os.path.join(saving_graph_path, 'unimib_Walking_umap'))
# print("Generate Walking done!\n")

# # Jumping
# syn_Jumping = generate_synthetic_signals(ckpt, 1500, class_name = 'Jumping')
# with open(os.path.join(saving_data_path, 'syn_Jumping.npy'), 'wb') as f:
#     np.save(f, syn_Jumping)
# visualization.visualization(unimib_Jumping.one_class_train_data, syn_Jumping, 'umap', os.path.join(saving_graph_path, 'unimib_Jumping_umap'))
# print("Generate Jumping done!\n")

# # SittingDown
# syn_SittingDown = generate_synthetic_signals(ckpt, 1500, class_name = 'SittingDown')
# with open(os.path.join(saving_data_path, 'syn_SittingDown.npy'), 'wb') as f:
#     np.save(f, syn_SittingDown)
# visualization.visualization(unimib_SittingDown.one_class_train_data, syn_SittingDown, 'umap', os.path.join(saving_graph_path, 'unimib_SittingDown_umap'))
# print("Generate SittingDown done!\n")

# # StandingUpFS
# syn_StandingUpFS = generate_synthetic_signals(ckpt, 1500, class_name = 'StandingUpFS')
# with open(os.path.join(saving_data_path, 'syn_StandingUpFS.npy'), 'wb') as f:
#     np.save(f, syn_StandingUpFS)
# visualization.visualization(unimib_StandingUpFS.one_class_train_data, syn_StandingUpFS, 'umap', os.path.join(saving_graph_path, 'unimib_StandingUpFS_umap'))
# print("Generate StandingUpFS done!\n")

# # StandingUpFL
# syn_StandingUpFL = generate_synthetic_signals(ckpt, 1500, class_name = 'StandingUpFL')
# with open(os.path.join(saving_data_path, 'syn_StandingUpFL.npy'), 'wb') as f:
#     np.save(f, syn_StandingUpFL)
# visualization.visualization(unimib_StandingUpFL.one_class_train_data, syn_StandingUpFL, 'umap', os.path.join(saving_graph_path, 'unimib_StandingUpFL_umap'))
# print("Generate StandingUpFL done!\n")

# Running
syn_Running = generate_synthetic_signals(ckpt, 1600, class_name = 'Running')
with open(os.path.join(saving_data_path, 'syn_Running.npy'), 'wb') as f:
    np.save(f, syn_Running)
visualization.visualization(unimib_Running.one_class_train_data, syn_Running, 'umap', os.path.join(saving_graph_path, 'unimib_Running_umap'))
print("Generate Running done!\n")

# GoingUpS
syn_GoingUpS = generate_synthetic_signals(ckpt, 1500, class_name = 'GoingUpS')
with open(os.path.join(saving_data_path, 'syn_GoingUpS.npy'), 'wb') as f:
    np.save(f, syn_GoingUpS)
visualization.visualization(unimib_GoingUpS.one_class_train_data, syn_GoingUpS, 'umap', os.path.join(saving_graph_path, 'unimib_GoingUpS_umap'))
print("Generate GoingUpS done!\n")

# GoingDownS
syn_GoingDownS = generate_synthetic_signals(ckpt, 1500, class_name = 'GoingDownS')
with open(os.path.join(saving_data_path, 'syn_GoingDownS.npy'), 'wb') as f:
    np.save(f, syn_GoingDownS)
visualization.visualization(unimib_GoingDownS.one_class_train_data, syn_GoingDownS, 'umap', os.path.join(saving_graph_path, 'unimib_GoingDownS_umap'))
print("Generate GoingDownS done!\n")

# LyingDownFS
syn_LyingDownFS = generate_synthetic_signals(ckpt, 1500, class_name = 'LyingDownFS')
with open(os.path.join(saving_data_path, 'syn_LyingDownFS.npy'), 'wb') as f:
    np.save(f, syn_LyingDownFS)
visualization.visualization(unimib_LyingDownFS.one_class_train_data, syn_LyingDownFS, 'umap', os.path.join(saving_graph_path, 'unimib_LyingDownFS_umap'))
print("Generate LyingDownFS done!\n")
