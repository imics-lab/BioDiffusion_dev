import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules1D_cond import Unet1D, GaussianDiffusion1D
import logging
from torch.utils.tensorboard import SummaryWriter
from MNIST import *
from torch.utils import data

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    mnist = MNIST_Conditional(oneD = True)
    dataloader = data.DataLoader(mnist, batch_size=32, num_workers=4, shuffle=True)
    model = Unet1D(
        dim = 64,
        self_condition = True,
        dim_mults = (1, 2, 4, 8),
        channels = 1).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 28*28,
        timesteps = 1000,
        objective = 'pred_v').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    sample_size = 10
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, dic in enumerate(pbar):
            org_img = dic['org_img'].to(device).to(torch.float)
            org_label = dic['org_label']
            target_img = dic['target_img'].to(device).to(torch.float)
            target_label = dic['target_label']
            cond_label = dic['cond_label']
            
            loss = diffusion(org_img, target_img) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)
        
        
        index_list = [i for i in range(len(mnist))]
        random.shuffle(index_list)
        cond_data = mnist.tagt_data[index_list][:sample_size].to(device).to(torch.float)
        sampled_signals = diffusion.sample(input_cond =cond_data , batch_size = sample_size)
        sampled_signals.shape # (10, 1, 28*28)
       
        
        is_best = False
        
        save_images_1D_to_2D(sampled_signals, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM1D_condtional_MNIST"
    args.epochs = 500
    args.batch_size = 32
    args.device = "cuda:7"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
