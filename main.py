#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Wed Sep 19 20:30:48 2018
Info:
References: https://github.com/pytorch/examples/tree/master/imagenet
'''
import argparse
import os
import random
import time
import datetime
import math
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from tensorboardX import SummaryWriter

from carla_net import CarlaNet
from carla_loader import CarlaH5Data
from helper import AverageMeter, save_checkpoint

parser = argparse.ArgumentParser(description='Carla CIL training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=10, type=int, metavar='N',
                    help='batch size of training')
parser.add_argument('--speed_weight', default=1, type=float,
                    help='speed weight')
parser.add_argument('--branch-weight', default=1, type=float,
                    help='branch weight')
parser.add_argument('--id', default="18101900", type=str)
parser.add_argument('--train-dir', default="/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/chosen_weather_train/clearnoon_h5/",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--eval-dir', default="/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/chosen_weather_test/clearnoon_h5/",
                    type=str, metavar='PATH',
                    help='evaluation dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--img-width', default=1216, type=int,
                    help='initial learning rate')
parser.add_argument('--img-height', default=368, type=int,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-log', default="",
                    type=str, metavar='PATH',
                    help='path to log evaluation results (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def log_args(logger):
    '''
    log args
    '''
    attrs = [(p, getattr(args, p)) for p in dir(args) if not p.startswith('_')]
    for key, value in attrs:
        output_log("{}: {}".format(key, value), logger=logger)


def main():
    global args
    args = parser.parse_args()
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"),
                        level=logging.ERROR)
    tsbd = SummaryWriter(log_dir=run_dir)
    log_args(logging)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        output_log(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.', logger=logging)

    if args.gpu is not None:
        output_log('You have chosen a specific GPU. This will completely '
                   'disable data parallelism.', logger=logging)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=0)

    model = CarlaNet()
    # criterion = EgoLoss()
    criterion = nn.MSELoss()

    tsbd.add_graph(model,
                   (torch.zeros(1, 3, 88, 200),
                    torch.zeros(1, 1)))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # TODO check other papers optimizers
    optimizer = optim.Adam(
        model.parameters(), args.lr, betas=(0.7, 0.85))
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint '{}'".format(args.resume),
                       logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume),
                       logging)

    cudnn.benchmark = True

    carla_data = CarlaH5Data(
        train_folder=args.train_dir,
        eval_folder=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.workers)

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]
    best_prec = math.inf

    if args.evaluate:
        args.id = args.id+"_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'"
                       .format(args.resume), logging)
            return
        # TODO here we should load test dataset
        if args.evaluate_log == "":
            output_log("=> please set evaluate log path with --evaluate-log <log-path>")
        evaluate(eval_loader, model, criterion,
                 os.path.join(log_dir, args.evaluate_log))
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()
        branch_losses, speed_losses, losses = \
            train(train_loader, model, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'optimizer': optimizer.state_dict()},
            args.id,
            is_best,
            os.path.join(
                save_weight_dir,
                "{}_{}.pth.tar".format(epoch+1, args.id))
            )


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    step = epoch * len(loader)
    for i, (img, speed, target, mask) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img = img.cuda(args.gpu, non_blocking=True)
        speed = speed.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        mask = mask.cuda(args.gpu, non_blocking=True)

        branches_out, pred_speed = model(img, speed)

        mask_out = branches_out * mask
        branch_loss = criterion(mask_out, target)
        speed_loss = criterion(pred_speed, speed)

        loss = args.branch_weight * branch_loss + \
            args.speed_weight * speed_loss

        losses.update(loss.item(), args.batch_size)
        branch_losses.update(branch_loss.item(), args.batch_size)
        speed_losses.update(speed_loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            writer.add_scalar('train/branch_loss', branch_losses.val, step+i)
            writer.add_scalar('train/speed_loss', speed_losses.val, step+i)
            writer.add_scalar('train/loss', losses.val, step+i)
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Branch loss {branch_loss.val:.3f} ({branch_loss.avg:.3f})\t'
                'Speed loss {speed_loss.val:.3f} ({speed_loss.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, branch_loss=branch_losses,
                    speed_loss=speed_losses, loss=losses), logging)

        step += 1
    return branch_losses.avg, speed_losses.avg, losses.avg


def evaluate(loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.gpu, non_blocking=True)
            speed = speed.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)

            branches_out, pred_speed = model(img, speed)

            mask_out = branches_out * mask
            branch_loss = criterion(mask_out, target)
            speed_loss = criterion(pred_speed, speed)

            loss = args.branch_weight * branch_loss + \
                args.speed_weight * speed_loss

            # measure accuracy and record loss
            losses.update(loss.item(), args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output_log(
                  'Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      i, len(loader), batch_time=batch_time,
                      loss=losses), logging)

    return losses.avg


# def evaluate(eval_loader, model, criterion, eval_res_path):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#     list_egomotions = []
#     with torch.no_grad():
#         end = time.time()
#         for i, (frame_0, frame_1, _) in enumerate(eval_loader):
#             if args.gpu is not None:
#                 for j in frame_0.keys():
#                     frame_0[j] = frame_0[j].cuda(args.gpu, non_blocking=True)
#                     frame_1[j] = frame_1[j].cuda(args.gpu, non_blocking=True)
#
#             # compute output
#             egomotions = model(
#                 torch.cat([frame_0['lidar'], frame_1['lidar']],dim=0),
#                 torch.cat([frame_1['lidar'], frame_0['lidar']],dim=0)
#                 )
#
#             loss = criterion(
#                 egomotions,
#                 torch.cat([frame_0['ego'], frame_1['ego']],dim=0)
#                 )
#
#             # measure accuracy and record loss
#             losses.update(loss.item(), args.batch_size)
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#             list_egomotions += [ np.array2string(egomotions[i, ::].data.cpu().numpy(), formatter={'float_kind':lambda x: "%.4f" % x})[1:-1] for i in range(args.batch_size)]
#             if i % args.print_freq == 0:
#                 output_log('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
#                        i, len(eval_loader), batch_time=batch_time, loss=losses), logging)
#
#     with open(eval_res_path, 'w') as f:
#         for _, line in enumerate(list_egomotions):
#             f.write(line+'\n')
#     return losses.avg

if __name__ == '__main__':
    main()
