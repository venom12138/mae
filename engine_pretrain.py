# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.utils import AverageMeter

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, exp=None,
                    args=None):
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    train_batches_num = len(data_loader)
    model.train(True)

    optimizer.zero_grad()
    end = time.time()

    for data_iter_step, (samples, _) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.cuda()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=True)

        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss_value, samples.size(0))

        if (i + 1) % 20 == 0:
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                epoch, data_iter_step + 1, train_batches_num, batch_time=batch_time,
                loss=losses))

            exp.log(string)

    return OrderedDict(loss=losses.ave)