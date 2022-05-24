# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import timm
import torch.nn.functional as F
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from networks import models_mae

from engine_pretrain import train_one_epoch
from util.utils import ExpHandler
from collections import OrderedDict
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_deit_tiny_patch4_dec512d', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=None, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../data', type=str,
                        help='dataset path')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--phase', type=str, default=None, choices=['pretrain', 'linprobe'])
    parser.add_argument('--en_wandb', action='store_true')
    return parser.parse_args()

args = get_args_parser()
exp = ExpHandler(en_wandb=args.en_wandb, args=args)
exp.save_config(args)

def main():
    global args, exp

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    transform_test = transforms.Compose([
        transforms.ToTensor(),])
    
    kwargs = {'num_workers': 10, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['cifar10'.upper()]('../data', train=True, download=True,
                                                transform=transform_test),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.cuda()

    model_without_ddp = model
    model = torch.nn.DataParallel(model)
    print("Model = %s" % str(model_without_ddp))   

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
    
    import matplotlib.pyplot as plt

    for data_iter_step, (samples, _) in enumerate(train_loader):
        samples = samples.cuda()

        with torch.cuda.amp.autocast():
            _, pred, _ = model(imgs=samples, mask_ratio=args.mask_ratio)
            img = model.module.unpatchify(pred).squeeze()
        
        img = transform_convert(img,transform_test)
        raw = transform_convert(samples.squeeze(),transform_test)
        # print(img.size())
        plt.imshow(raw)
        plt.savefig(f'../imgs/{data_iter_step}-raw.png')
        plt.imshow(img)
        plt.savefig(f'../imgs/{data_iter_step}.png')

def transform_convert(img_tensor,transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])

    img_tensor = img_tensor.transpose(0,2).transpose(0,1).to('cpu')  # C x H x W  ---> H x W x C
    
    img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.numpy()
    
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img

if __name__ == '__main__':
    
    main()
