import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
from datetime import datetime
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
import models as models

from sdd import sdd
from sdd_rr import sdd_rr

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('-s', '--save-path', default='', type=str, metavar='DIR',
                    help='path to save')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
ffn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global prev_feat, conv_feat
def prev_hook(module, inputdata, outputdata):
    global prev_feat
    prev_feat = outputdata.data#.cpu().numpy()
def conv_hook(module, inputdata, outputdata):
    global conv_feat
    conv_feat = outputdata.data#.cpu()#.numpy()

def main():
    args = parser.parse_args()

    if not args.save_path:
        args.save_path = args.arch
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model_ffn = models.__dict__[args.arch]()
    model_pretrained = models.__dict__[args.arch]()
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        state_dict = torch.load(args.pretrained)
        model_ffn.load(state_dict)
        model_pretrained.load(state_dict)
    else:
        assert(False)

    model_ffn = model_ffn.cuda()
    model_pretrained = model_pretrained.cuda(1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model_ffn.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    torch.backend.cudnn.enabled = False

    if 'alexnet' in args.arch:
        input_size = 227
    else:
        input_size = 224


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_pretrained.eval()
    model_ffn.eval()

    conv_idx = 0
    for layer_idx in range(len(model_pretrained.features)):
        module = model_pretrained.features[layer_idx]
        if isinstance(module, nn.Conv2d): 
            if conv_idx > 0:
                # layers
                conv = module
                conv_ffn = model_ffn.features[layer_idx]
                prev = model_ffn.features[layer_idx-1]

                # conv parameters
                kernel_h, kernel_w = conv.kernel_size
                pad_h, pad_w = conv.padding
                stride_h, stride_w = conv.stride

                handle_prev = prev.register_forward_hook(prev_hook)
                handle_conv = conv.register_forward_hook(conv_hook)

                batch_iterator = iter(train_loader)

                # weights and bias
                W = conv.weight.data#.cpu()
                bias = conv.bias.data#.cpu()
                print(W.shape)

                # feat extract
                max_items = 8000
                input, target = next(batch_iterator)
                input_pretrained = input.cuda(device=pretrained_device, non_blocking=True)
                input_ffn = input.cuda(device=ffn_device, non_blocking=True)
                model_pretrained(input_pretrained)
                model_ffn(input_ffn)
                print(prev_feat.shape)
                print(conv_feat.shape)
                [prev_feat_n, prev_feat_c, prev_feat_h, prev_feat_w] = prev_feat.shape
                [conv_feat_n, conv_feat_c, conv_feat_h, conv_feat_w] = conv_feat.shape
                X = torch.zeros(max_items, prev_feat_c*kernel_h*kernel_w).to(ffn_device)
                Y = torch.zeros(max_items, conv_feat_c).to(pretrained_device)
                print(X.shape)
                print(Y.shape)

                n_items = 0
                for batch_idx in range(0, max_items):
                    input, target = next(batch_iterator)
                    input_pretrained = input.cuda(device=pretrained_device, non_blocking=True)
                    model_pretrained(input_pretrained)
                    input_ffn = input.cuda(device=ffn_device, non_blocking=True)
                    model_ffn(input_ffn)
                
                    prev_feat_pad = torch.zeros(prev_feat_n, prev_feat_c, prev_feat_h+2*pad_h, prev_feat_w+2*pad_w).to(ffn_device)
                    prev_feat_pad[:, :, pad_h:pad_h+prev_feat_h, pad_w:pad_w+prev_feat_w] = prev_feat
                    prev_feat_pad = prev_feat_pad.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w).permute(0,2,3,1,4,5)
                    [feat_pad_n, feat_pad_h, feat_pad_w, feat_pad_c, feat_pad_hh, feat_pad_ww] = prev_feat_pad.shape
                    prev_feat_pad = prev_feat_pad.reshape(feat_pad_n*feat_pad_h*feat_pad_w, -1)
                    conv_feat_tmp = conv_feat.permute(0,2,3,1).reshape(-1, conv_feat_c) - bias

                    n_positions = prev_feat_pad.shape[0]
                    if n_items + n_positions >= max_items:
                        n_positions = max_items - n_items
                        X[n_items:n_items+n_positions] = prev_feat_pad
                        Y[n_items:n_items+n_positions] = conv_feat_tmp
                        break
                    else:
                        X[n_items:n_items+n_positions] = prev_feat_pad
                        Y[n_items:n_items+n_positions] = conv_feat_tmp

                handle_prev.remove()
                handle_conv.remove()

                ## sdd init
                W_shape = W.shape
                W = W.reshape(W_shape[0], -1)
                [m, n] = W.shape
                kmax = min(m, n)

                print('run sdd')
                start_time = time.time()
                D, U, V = sdd(W.cpu().numpy(), kmax)
                end_time = time.time()
                print('Time of ssd for conv'+str(conv_idx+1)+'is', end_time-start_time, 's')
                # save
                with open(args.save_path+'/conv'+ str(conv_idx+1) + '_' + str(kmax) + '.pkl', 'wb') as f:
                    pickle.dump({'D': D, 'U': U, 'V': V}, f, pickle.HIGHEST_PROTOCOL)
        
                print('run sdd_rr')
                start_time = time.time()
                D, U, V = sdd_rr(X.cpu().numpy().T, Y.cpu().numpy().T, D, U, V, max_epoch=50)
                end_time = time.time()
                print('Time of ssd_rr for conv'+str(conv_idx+1)+'is', end_time-start_time, 's')
                # save
                with open(args.save_path+'/conv'+ str(conv_idx+1) + '_' + str(kmax) + '_rr_e50.pkl', 'wb') as f:
                    pickle.dump({'D': D, 'U': U, 'V': V}, f, pickle.HIGHEST_PROTOCOL)

                ## sdd update using Feature-map Reconstruction
                W_r = np.dot(np.multiply(U, D), V.T)
                W_r = W_r.reshape(W_shape)
                conv_ffn.weight.data.copy_(torch.from_numpy(W_r))
                print('validation')
                validate(val_loader, model_ffn, criterion, args)
            conv_idx = conv_idx + 1

    upbn(train_loader, model_ffn, criterion, optimizer, args, 1000)
    validate(val_loader, model_ffn, criterion, args)
    save_state_dict(model_ffn.state_dict(), args.save_path, 'alexnet_bn_conv_e50.pth')

    print('Processing FC layers')
    fc_idx = 0
    for layer_idx in range(len(model_pretrained.classifier)):
        module = model_pretrained.classifier[layer_idx]
        if isinstance(module, nn.Linear): 
            # layers
            conv = module
            # weights and bias
            W = conv.weight.data#.cpu()
            bias = conv.bias.data#.cpu()

            ## sdd init
            W_shape = W.shape
            W = W.reshape(W_shape[0], -1)
            [m, n] = W.shape
            kmax = 3072
            print('run sdd')
            start_time = time.time()
            D, U, V = sdd(W.cpu().numpy(), kmax)
            end_time = time.time()
            print('Time of ssd for fc'+str(6+fc_idx)+'is', end_time-start_time, 's')
            with open(args.save_path+'/fc'+ str(6+fc_idx) + '_' + str(kmax) + '.pkl', 'wb') as f:
                    pickle.dump({'D': D, 'U': U, 'V': V}, f, pickle.HIGHEST_PROTOCOL)

            W_r = np.dot(np.multiply(U, D), V.T)
            W_r = W_r.reshape(W_shape)
            conv_ffn.weight.data.copy_(torch.from_numpy(W_r))
            print('validation')
            validate(val_loader, model_ffn, criterion, args)
            if fc_idx == 1:
                break
            fc_idx = fc_idx + 1
            #break

    upbn(train_loader, model_ffn, criterion, optimizer, args, 1000)
    validate(val_loader, model_ffn, criterion, args)
    save_state_dict(model_ffn.state_dict(), args.save_path, 'alexnet_bn_conv_fc_e50.pth')


def upbn(train_loader, model, criterion, optimizer, args, max_iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    global_iter = 0
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            global_iter = global_iter + 1
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       0, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            if global_iter == max_iter:
                break

def save_state_dict(state_dict, path, filename='state_dict.pth'):
    saved_path = os.path.join(path, filename)
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if '.module.' in key:
            new_state_dict[key.replace('.module.', '.')] = state_dict[key].cpu()
        else:
            new_state_dict[key] = state_dict[key].cpu()
    torch.save(new_state_dict, saved_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
