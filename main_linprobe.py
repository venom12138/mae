# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

from networks import models_vit

from engine_finetune import train_one_epoch, evaluate
from util.utils import ExpHandler
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch4_32', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    # parser.set_defaults(cls_token=True)

    # Dataset parameters
    parser.add_argument('--data_path', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--phase', type=str, default=None, choices=['pretrain', 'linprobe', 'finetune'])
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

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
    kwargs = {'num_workers': 10, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['cifar10'.upper()]('../data', train=True, download=True,
                                                transform=transform_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['cifar10'.upper()]('../data', train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.cuda()

    model_without_ddp = model
    model = torch.nn.DataParallel(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

    if args.eval:
        test_stats = evaluate(val_loader, model, exp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            max_norm=None,
            exp=exp,
            args=args)
        save_checkpoint({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint=exp.save_dir, filename=f'checkpoint.pth.tar')

        eval_metrics = evaluate(val_loader, model, exp, epoch)
        max_accuracy = max(max_accuracy, eval_metrics["top1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        exp.write(phase, train_metrics=train_metrics, eval_metrics=eval_metrics,epoch=epoch,lr=optimizer.param_groups[0]['lr'])
        
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
