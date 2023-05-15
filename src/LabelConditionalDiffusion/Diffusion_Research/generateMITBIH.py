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
from MITBIH import *
import visualization
import torch
import os
from modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free

############## load devices #################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############## load real dataset #############
# load MITBIH dataset
data_path = "/home/x_l30/Diffusion/Diffusion_Research/datasets/MITBIH/mitbih_train.csv"
mitbih_cls0 = mitbih_oneClass(filename = data_path, class_id=0)
mitbih_cls1 = mitbih_oneClass(filename = data_path, class_id=1)
mitbih_cls2 = mitbih_oneClass(filename = data_path, class_id=2)
mitbih_cls3 = mitbih_oneClass(filename = data_path, class_id=3)
mitbih_cls4 = mitbih_oneClass(filename = data_path, class_id=4)



############# load MITBIH diffusion model
def generate_synthetic_signals(checkpoint, num_samples = 100, class_id = 0):
    model = Unet1D_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = 5,
        cond_drop_prob = 0.5,
        channels = 1).to(device)

    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length = 128,
        timesteps = 1000).to(device)
    
    ckp = torch.load(checkpoint)
    model.load_state_dict(ckp['model_state_dict'])
    
    y = torch.Tensor([class_id] * num_samples).long().to(device)
    x = diffusion.sample(classes = y, cond_scale = 3.)

    sampled_signals = x.to('cpu').detach().numpy()
    print(sampled_signals.shape)
    return sampled_signals

current_path = os.getcwd()
print(f"Current path: {current_path}")

saving_graph_path = './synthetic/visualizations/'
saving_data_path = './synthetic/synthetic_data/'

ckpt = "./checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt"

############# generate synthetic MITBIH data ############

#Class 0 'Non-Ectopic Beats'
syn_class0 = generate_synthetic_signals(ckpt, 7000, class_id = 0)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class0.npy'), 'wb') as f:
    np.save(f, syn_class0)
visualization.visualization (mitbih_cls0.data[:1000], syn_class0, 'umap', os.path.join(saving_graph_path, 'MITBIH_class0_umap'))
print("Generate class 0 done!\n")

# Class 1 'Superventrical Ectopic'
syn_class1 = generate_synthetic_signals(ckpt, 7000, class_id = 1)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class1.npy'), 'wb') as f:
    np.save(f, syn_class1)
visualization.visualization(mitbih_cls1.data[:1000], syn_class1, 'umap', os.path.join(saving_graph_path, 'MITBIH_class1_umap'))
print("Generate class 1 done!\n")


# Class 2 'Ventricular Beats'
syn_class2 = generate_synthetic_signals(ckpt, 7000, class_id = 2)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class2.npy'), 'wb') as f:
    np.save(f, syn_class2)
visualization.visualization(mitbih_cls2.data[:1000], syn_class2, 'umap', os.path.join(saving_graph_path, 'MITBIH_class2_umap'))
print("Generate class 2 done!\n")

# Class 3 'Unknown'
syn_class3 = generate_synthetic_signals(ckpt, 7000, class_id = 3)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class3.npy'), 'wb') as f:
    np.save(f, syn_class3)
visualization.visualization(mitbih_cls3.data, syn_class3, 'umap', os.path.join(saving_graph_path, 'MITBIH_class3_umap'))
print("Generate class 3 done!\n")

# Class 4 'Fusion Beats'
syn_class4 = generate_synthetic_signals(ckpt, 7000, class_id = 4)
with open(os.path.join(saving_data_path, 'syn_MITBIH_class4.npy'), 'wb') as f:
    np.save(f, syn_class4)
visualization.visualization(mitbih_cls4.data[:1000], syn_class4, 'umap', os.path.join(saving_graph_path, 'MITBIH_class4_umap'))
print("Generate class 4 done!\n")
