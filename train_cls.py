from __future__ import print_function,division

import os
import sys
sys.path.append('./utils/')

import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from model.pointnet2_cls import pointnet2_cls
from data_utils import pts_cls_dataset,pts_collate


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet')
parser.add_argument('--data', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/modelnet/test_files.txt',
                    help='txt file to dataset')
parser.add_argument('--data-eval', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/modelnet/test_files.txt',
                    help='txt file to validate dataset')
parser.add_argument('--log', metavar='LOG',default='log.txt',
                    help='filename of log file')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-step', default=500, type=int, metavar='N',
                    help='number of iter to write log')
parser.add_argument('--test-step', default=1000, type=int, metavar='N',
                    help='number of iter to evaluate ')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs',  '--batch-size', default=2 , type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--decay_step', default=200000, type=int,
                    metavar='LR', help='decay_step of learning rate')
parser.add_argument('--decay_rate', default=0.7, type=float,
                    metavar='LR', help='decay_rate of learning rate')

parser.add_argument('--fps', action='store_true', help='Whether to use fps')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--resume', default='cls_checkpoint.pth',type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()
logname=args.log

if is_GPU:
    torch.cuda.set_device(args.gpu)


my_dataset=pts_cls_dataset(datalist_path=args.data,use_extra_feature=args.normal)
data_loader = torch.utils.data.DataLoader(my_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=pts_collate)

net=pointnet2_cls(input_dim=3,use_FPS=args.fps)
if is_GPU:
    net=net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
critenrion=nn.NLLLoss()

def save_checkpoint(epoch,model,num_iter):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'iter':num_iter,
    },args.resume)

def log(filename,epoch,batch,loss):
    f1=open(filename,'a')
    if epoch == 0 and batch == 0:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    f1.write('\nin epoch{} batch{} loss={} '.format(epoch,batch,loss))
    f1.close()

def evaluate(model_test):
    model_test.eval()
    total_correct=0

    data_eval = pts_cls_dataset(datalist_path=args.data_eval,data_argument=False)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=4, shuffle=True, collate_fn=pts_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx, (pts, label) in enumerate(eval_loader):
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
        pred = net(pts)

        _, pred_index = torch.max(pred, dim=1)
        num_correct = (pred_index.eq(label)).data.cpu().sum().item()
        total_correct +=num_correct

    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))

    model_test.train()
    with open(logname,'a') as f:
        f.write('\nthe evaluate average accuracy:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))

def train():

    net.train()
    num_iter=0
    start_epoch=0

    if os.path.exists(args.resume):
        if is_GPU:
            checkoint = torch.load(args.resume)
        else:
            checkoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        start_epoch = checkoint['epoch']
        net.load = net.load_state_dict(checkoint['model'])
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("Warining! No resume checkpoint to load")

    print('start training')

    for epoch in xrange(start_epoch,args.epochs):
        init_epochtime = time.time()

        for batch_idx, (pts, label) in enumerate(data_loader):
            t1=time.time()
            if is_GPU:
                pts = Variable(pts.cuda())
                label = Variable(label.cuda())
            else:
                pts = Variable(pts)
                label = Variable(label)

            pred = net(pts)
            loss = critenrion(pred, label)

            _, pred_index = torch.max(pred, dim=1)
            num_correct = (pred_index.eq(label)).data.cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2=time.time()
            num_iter+=1

            print('In Epoch{} Iter{},loss={} accuracy={}  time cost:{}'.format(epoch,num_iter, loss.data,num_correct.item() / args.batch_size,t2-t1))
            if num_iter%(args.log_step*10)==0 and num_iter!=0:
                save_checkpoint(epoch, net, num_iter)
                evaluate(net)
            if num_iter%(args.log_step)==0 and num_iter!=0:
                log(logname, epoch, num_iter, loss.data)

            if (num_iter*args.batch_size)%args.decay_step==0 and num_iter!=0:
                f1 = open(logname, 'a')
                f1.write("learning rate decay in iter{}\n".format(num_iter))
                f1.close()
                print ("learning rate decay in iter{}\n".format(num_iter))
                for param in optimizer.param_groups:
                    param['lr'] *= args.decay_rate
                    param['lr'] = max(param['lr'],0.00001)

        end_epochtime = time.time()
        print('--------------------------------------------------------')
        print('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print('-------------------------------------------------------- \n')

    save_checkpoint(args.epochs-1, net, num_iter)
    evaluate(net)

train()
