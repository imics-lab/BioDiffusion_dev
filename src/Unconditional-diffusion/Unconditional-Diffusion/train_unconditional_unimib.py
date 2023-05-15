#!/usr/bin/env bash

import os
import argparse
import os

os.system(f"python Unconditional_UNIMIB.py \
--run-name 'DDPM1D_Uncondtional_UNIMIB_LyingDownFS' \
--epochs 100 \
--batch-size 32 \
--num-workers 4 \
--device 'cuda:7' \
--lr 3e-4 \
--data-path '/home/x_l30/Research/datasets/UniMiB/UniMiB-SHAR/data' \
--channels 3 \
--seq-length 144 \
--timesteps 1000 \
--single-class \
--class-name LyingDownFS \
")