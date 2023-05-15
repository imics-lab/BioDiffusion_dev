#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualization.py

Visualization metrics

Author:
Date: 


TODOS:
* add multi-class umap
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
   
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

        ax.legend(fontsize=14)  
        # plt.title('PCA plot')
        plt.xlabel('x-pca', fontsize=14)
        plt.ylabel('y_pca', fontsize=14)
        # plt.show()

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

        ax.legend(fontsize=14)

        # plt.title('t-SNE plot')
        plt.xlabel('x-tsne', fontsize=14)
        plt.ylabel('y_tsne', fontsize=14)
        # plt.show()    

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
        
        ax.legend(fontsize=14)

        # plt.title('UMAP plot')
        plt.xlabel('x-umap', fontsize=14)
        plt.ylabel('y_umap', fontsize=14)

        
    plt.savefig(f'./{save_name}.pdf', format="pdf")
    # When you call plt.show() before plt.savefig(), the figure is displayed using the backend specified in your Matplotlib configuration. After displaying the figure, the current figure is closed and reset, causing the saved figure to be empty.
    plt.show()
    plt.close()




def draw_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='UMAP projection', random_state=42):
    """
    Draw UMAP of a 3-channel signal dataset with separate colors for each class.

    Args:
    data: numpy array, the dataset to be visualized with shape (number_of_samples, channels, length)
    labels: numpy array, class labels corresponding to each data point
    n_neighbors: int, optional, default: 15, number of neighbors to consider for each point in UMAP
    min_dist: float, optional, default: 0.1, minimum distance between points in the low-dimensional representation
    n_components: int, optional, default: 2, number of dimensions in the low-dimensional representation
    metric: str, optional, default: 'euclidean', distance metric to use for UMAP
    title: str, optional, default: 'UMAP projection', title of the plot
    """

    # Reshape the data by concatenating channels along columns
    number_of_samples, channels, length = data.shape
    reshaped_data = data.reshape(number_of_samples, channels * length)

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(reshaped_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette=sns.color_palette("hls", len(np.unique(labels))), edgecolor='k', s=100, ax=ax)

    # Set plot title and labels
    # ax.set_title(title, fontsize=18)
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    
    plt.savefig(f'./{title}.pdf')
    # Show the plot
    plt.show()