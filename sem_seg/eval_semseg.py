from __future__ import print_function,division

import os
import sys
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../')
from pointnet_semseg import pointnet2_semseg
from s3dis_utils import indoor3d_dataset,pts_collate_seg

is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet_partseg')
parser.add_argument('--data', metavar='DIR',
                    default='../../indoor3d_sem_seg_hdf5_data/all_files.txt',help='txt file to dataset')
parser.add_argument('--area-eval', metavar='N',
                    default='Area_6',help='validate area dataset')
parser.add_argument('--log', metavar='LOG',
                    default='log_partsegmentation',help='dir of log file and resume')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('-bs',  '--batch-size', default=2 , type=int,
                    metavar='N', help='mini-batch size (default: 2)')

parser.add_argument('--fps', action='store_true', help='Whether to use fps')
parser.add_argument('--color', action='store_true', help='Whether to use color')
parser.add_argument('--resume', default=None,type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()


NUM_POINTS = 4096
EVAL_BATCHSIZE = args.batch_size
RESUME = args.resume

data_eval = indoor3d_dataset(args.data, test_area=args.area_eval, training=False, use_color=args.color)
eval_loader = torch.utils.data.DataLoader(data_eval,num_workers=4,
            batch_size=EVAL_BATCHSIZE, shuffle=True, collate_fn=pts_collate_seg)

if args.color:
    feat_dim = 6
else:
    feat_dim = 3
net = pointnet2_semseg(input_dim=feat_dim, use_FPS=args.fps, num_class=13)
if is_GPU:
    net = net.cuda()
if os.path.exists(args.resume):
    if is_GPU:
        checkoint = torch.load(args.resume)
    else:
        checkoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    start_epoch = checkoint['epoch']
    net.load = net.load_state_dict(checkoint['model'])
    num_iter = checkoint['iter']
    print('load the resume checkpoint,train from epoch{}'.format(start_epoch))
else:
    print("Warining! No resume checkpoint to load")

def evaluate(model_test):
    NUM_CLASSES = 13

    model_test.eval()
    total_correct = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    print("dataset size:", len(eval_loader.dataset))

    for batch_idx, (pts, seg) in enumerate(eval_loader):
        ## pts [N,P,6] seg [N,P]
        if is_GPU:
            pts = Variable(pts.cuda())
            seg_label = Variable(seg.cuda())
        else:
            pts = Variable(pts)
            seg_label = Variable(seg)
        ## pred [N,13,P]
        pred = net(pts)

        _, pred_index = torch.max(pred, dim=1)  ##[N,P]
        pred_index, seg_label = pred_index.view(-1,), seg_label.view(-1,) ##[N*P,]
        num_correct = (pred_index.eq(seg_label)).data.cpu().sum()
        print('in batch{} acc={}'.format(batch_idx, num_correct.item() * 1.0 / (EVAL_BATCHSIZE * NUM_POINTS)))
        total_correct += num_correct.item()


        for idx,point_label in enumerate(seg_label):
            total_seen_class[point_label] += 1
            total_correct_class[point_label] += ((pred_index.eq(point_label))[idx]).item()

    model_test.train()
    print('the accuracy:{}'.format(total_correct * 1.0 / (len(eval_loader.dataset) * NUM_POINTS)))
    print('eval avg class acc: %f \n' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

evaluate(net)
