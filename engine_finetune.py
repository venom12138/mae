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
from typing import Iterable, Optional
import time
from collections import OrderedDict
import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.lr_sched as lr_sched
from util.utils import AverageMeter

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, exp=None,
                    args=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    train_batches_num = len(data_loader)
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    end = time.time()

    for data_iter_step, (samples, targets) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.cuda()
        targets = targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss_value, samples.size(0))
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        if (data_iter_step + 1) % accum_iter == 0:
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                epoch, data_iter_step + 1, train_batches_num, batch_time=batch_time,
                loss=losses))

            exp.log(string)

    return OrderedDict(loss=losses.ave)


@torch.no_grad()
def evaluate(data_loader, model, exp, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(data_loader)

    # switch to evaluation mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.cuda()
        target = target.cuda()

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output.data, target, topk=(1,))[0]

        batch_size = images.shape[0]
        losses.update(loss.data.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    
    string = ('Test: [{0}][{1}/{2}]\t'
            'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
            'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
            'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    exp.log(string)

    return OrderedDict(loss=losses.ave, top1=top1.ave)