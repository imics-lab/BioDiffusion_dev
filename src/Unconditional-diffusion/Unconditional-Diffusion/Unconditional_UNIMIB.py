'''
Unconditional UNIMIB Train and Test

Author:
Date:

'''

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
import modules
import datasets
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device    
    unimib = datasets.unimib_dataLoader(path_in = args.data_path, single_class = args.single_class, class_name= args.class_name)
    dataloader = data.DataLoader(unimib, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    model = modules.Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = args.channels).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = modules.GaussianDiffusion1D(
        model,
        seq_length = args.seq_length,
        timesteps = args.timesteps,
        objective = 'pred_v').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    start_epoch = 0
    if args.resume:
        ckp = torch.load(args.checkpoint)
        model.load_state_dict(ckp['model_state_dict'])
        start_epoch = ckp['epoch']
        
    l = len(dataloader)
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (signals, _) in enumerate(pbar):
            signals = signals.to(device).to(torch.float)
            loss = diffusion(signals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

        sampled_signals = diffusion.sample(batch_size = 10)
        sampled_signals.shape # (10, 3, 150)
        
        save_signals(sampled_signals, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join("checkpoint", args.run_name))
        if epoch % 10 == 0:
            save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, os.path.join("checkpoint", args.run_name), filename=f'checkpoint_epoch{epoch}.pt')
            
def test():
    pass


# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM1D_Uncondtional_UNIMIB_Walking"
#     args.epochs = 50
#     args.batch_size = 32
#     args.num_workers = 4
#     args.device = "cuda:0"
#     args.lr = 3e-4
#     args.data_path = '/home/x_l30/Research/datasets/UniMiB/UniMiB-SHAR/data'
#     args.channels = 3
#     args.seq_length = 144
#     args.timesteps = 1000
#     args.class_name = 'Walking'
#     args.single_class = True
#     args.checkpoint = ''
#     args.resume = False
#     train(args)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DDPM Model')
    
    parser.add_argument('--run-name', type=str, default='DDPM1D_Uncondtional_UNIMIB_Walking', help='Description of the run name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='/home/x_l30/Research/datasets/UniMiB/UniMiB-SHAR/data', help='Path to data')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in the data')
    parser.add_argument('--seq-length', type=int, default=144, help='Sequence length')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion model timesteps')
    parser.add_argument('--class-name', type=str, default='Walking', help='Name of the class in the dataset')
    parser.add_argument('--single-class', action='store_true', help='Whether to use only one class')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training')
    args = parser.parse_args()
    
    train(args)