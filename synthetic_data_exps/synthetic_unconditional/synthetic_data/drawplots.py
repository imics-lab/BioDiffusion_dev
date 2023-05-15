import numpy as np
import matplotlib.pyplot as plt
import os
import random

# define the classes and their corresponding numpy files
classes = {
    'GoingDownS': 'syn_GoingDownS.npy',
    'LyingDownFS': 'syn_LyingDownFS.npy',
    'MITBIH_class0': 'syn_MITBIH_class0.npy',
    'MITBIH_class1': 'syn_MITBIH_class1.npy',
    'MITBIH_class2': 'syn_MITBIH_class2.npy',
    'MITBIH_class3': 'syn_MITBIH_class3.npy',
    'MITBIH_class4': 'syn_MITBIH_class4.npy',
    'Running': 'syn_Running.npy',
    'simu_class0': 'syn_simu_class0.npy',
    'simu_class1': 'syn_simu_class1.npy',
    'simu_class2': 'syn_simu_class2.npy',
    'simu_class3': 'syn_simu_class3.npy',
    'simu_class4': 'syn_simu_class4.npy',
    'SittingDown': 'syn_SittingDown.npy',
    'StandingUpFL': 'syn_StandingUpFL.npy',
    'StandingUpFS': 'syn_StandingUpFS.npy'
}

# define the number of signals to randomly sample from each class
num_signals_to_sample = 3

# loop through each class and sample the signals
for class_name, file_name in classes.items():
    signals = np.load(file_name)
    dim = signals.shape[1]
    if signals.shape[0] < num_signals_to_sample:
        print(f"Not enough signals in class {class_name}")
        continue
    random_indexs = random.sample(range(len(signals)), num_signals_to_sample)
    sampled_signals = signals[random_indexs]
    
    # plot the sampled signals in a line
    fig, ax = plt.subplots(1, 3, figsize=(10,3))
    for i in range(num_signals_to_sample):
        for d in range(dim):
            ax[i].plot(sampled_signals[i][d][:])
    
    # save the plot with the same name as the numpy file
    fig.savefig(os.path.splitext(file_name)[0] + '.pdf')
    plt.close(fig)
