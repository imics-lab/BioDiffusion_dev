#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualization.py

Visualization metrics for original and generated time-series data comparsion

Author: Xiaomin Li, Texas State University
Date: 11/10/2022


TODOS:
* add umap
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np
   
def visualization (ori_data, generated_data, analysis, save_name):
    """Using PCA, tSNE, UMAP for generated and original data visualization.

    Args:
    - ori_data: original data, data shape = [N_o, seq_len, dim]
    - generated_data: generated synthetic data, data shape = [N_g, seq_len, dim] (N_o >= N_g)
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)]) # at most choose 1000 data samples 
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    
    # make original and generated data has the same comparing size
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  
    
    #average dimensions
    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["blue" for i in range(anal_sample_no)] + ["red" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, s= 10, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, s= 10, label = "Synthetic")

        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, s= 10, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, s= 10, label = "Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()    

    elif analysis == 'umap':
        # Do UMAP Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
        
        # UMAP Analysis
        reducer = umap.UMAP(random_state=42)
        reducer.fit(prep_data_final)
        umap_results = reducer.transform(prep_data_final)
        
        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(umap_results[:anal_sample_no, 0], umap_results[:anal_sample_no, 1], 
                    c = colors[:anal_sample_no], alpha = 0.2, s= 10, label = "Original")
        plt.scatter(umap_results[anal_sample_no:, 0], umap_results[anal_sample_no:, 1], 
                    c = colors[anal_sample_no:], alpha = 0.2, s= 10, label = "Synthetic")
        
        ax.legend()

        plt.title('UMAP plot')
        plt.xlabel('x-umap')
        plt.ylabel('y_umap')
        plt.show()
        
    plt.show()
    plt.savefig(f'./{save_name}.pdf', format="pdf")
