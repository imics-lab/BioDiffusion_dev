#!/usr/bin/env bash

import os
import argparse
import os

os.system(f"python Unconditional_MITBIH.py \
--run-name 'DDPM1D_Uncondtional_MITBIH_allclass' \
--epochs 100 \
--batch-size 32 \
--num-workers 4 \
--device 'cuda:5' \
--lr 3e-4 \
--data-path '/home/x_l30/Research/datasets/MITBIH/mitbih_train.csv' \
--channels 1 \
--seq-length 128 \
--timesteps 1000 \
--all-class")