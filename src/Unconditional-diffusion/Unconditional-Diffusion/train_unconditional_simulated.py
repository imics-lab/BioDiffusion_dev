#!/usr/bin/env bash

import os
import argparse
import os

os.system(f"python Unconditional_simulated.py \
--run-name 'DDPM1D_Uncondtional_simulated_allcls' \
--epochs 100 \
--batch-size 32 \
--num-workers 4 \
--device 'cuda:7' \
--lr 3e-4 \
--data-path '/home/x_l30/Research/datasets/simulated/' \
--channels 1 \
--seq-length 512 \
--timesteps 1000 \
--all-class")