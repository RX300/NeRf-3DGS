import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from nerf_train import *
from test_nerf_helpers import *


import load_blender as blld


if __name__ == "__main__":
    expname = "blender_paper_chair"
    basedir = "../dataset/nerf_synthetic-20230812T151944Z-001"
    datadir = "./nerf_synthetic/chair"
    dataset_type = "blender"

    no_batching = True

    use_viewdirs = True
    white_bkgd = True
    lrate_decay = 500

    N_samples = 64
    N_importance = 128
    N_rand = 1024

    precrop_iters = 500
    precrop_frac = 0.5

    half_res = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    parser=config_parser()
    args=parser.parse_args()
    #改变args的参数
    args.expname = expname
    args.basedir = basedir
    args.datadir = datadir
    args.dataset_type = dataset_type
    args.no_batching = no_batching
    args.use_viewdirs = use_viewdirs
    args.white_bkgd = white_bkgd
    args.lrate_decay = lrate_decay
    args.N_samples = N_samples
    args.N_importance = N_importance
    args.N_rand = N_rand
    args.precrop_iters = precrop_iters
    args.precrop_frac = precrop_frac
    args.half_res = half_res

    #打印args的参数
    print("=== Args ===")
    print(args.N_samples)

    test_get_rays()
    test_load_blender_data()
