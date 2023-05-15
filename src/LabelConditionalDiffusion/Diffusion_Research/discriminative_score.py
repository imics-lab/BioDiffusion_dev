"""

discriminative_score.py

modified from the Time-series Generative Adversarial Networks (TimeGAN) Codebase
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import numpy as np


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
    """Divide train and test data for both original and synthetic data.
  
    Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
      """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]      

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
    """Returns Maximum sequence length and each sequence length.

    Args:
    - data: original data

    Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))

    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

    Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]     

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb




class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        y_hat_logit = self.fc(h[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

    

def discriminative_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim)
    d_solver = optim.Adam(discriminator.parameters())

    bce_loss = nn.BCEWithLogitsLoss()

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb = torch.tensor(X_mb, dtype=torch.float32) 
        T_mb = torch.tensor(T_mb, dtype=torch.float32)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32)
        T_hat_mb = torch.tensor(T_hat_mb, dtype=torch.float32)

        # Train discriminator
        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)

        d_loss_real = bce_loss(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = bce_loss(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        d_solver.zero_grad()
        d_loss.backward()
        d_solver.step()

    # Test the performance on the testing set
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_t = torch.tensor(test_t, dtype=torch.float32)
    test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32)
    test_t_hat = torch.tensor(test_t_hat, dtype=torch.float32)

    y_pred_real_curr, _ = discriminator(test_x)
    y_pred_fake_curr, _ = discriminator(test_x_hat)

    y_pred_final = np.concatenate((y_pred_real_curr.detach().numpy(), y_pred_fake_curr.detach().numpy()), axis=0)
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, y_pred_final > 0.5)
    print(f'acc is {acc}')
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
