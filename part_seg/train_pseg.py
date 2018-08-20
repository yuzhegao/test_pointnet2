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
from model.pointnet2_seg import pointnet2_seg
from data_utils import shapenet_dataset,pts_collate_seg


is_GPU=torch.cuda.is_available()

parser = argparse.ArgumentParser(description='pointnet_partseg')
parser.add_argument('--data', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/hdf5_data/test_hdf5_file_list.txt',
                    help='txt file to dataset')
parser.add_argument('--data-eval', metavar='DIR',default='/home/gaoyuzhe/Downloads/3d_data/hdf5_data/test_hdf5_file_list.txt',
                    help='txt file to validate dataset')
parser.add_argument('--log', metavar='LOG',default='log',
                    help='dir of log file and resume')

parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='the index  of GPU where program run')
parser.add_argument('--epochs', default=201, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-step', default=500, type=int, metavar='N',
                    help='number of iter to write log')
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
parser.add_argument('--resume', default='pointnet_partseg.pth',
                    type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()

LOG_DIR=args.log
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
resume=os.path.join(LOG_DIR,args.resume)
logname=os.path.join(LOG_DIR,'log.txt')

if is_GPU:
    torch.cuda.set_device(args.gpu)

my_dataset=shapenet_dataset(args.data)
data_loader=torch.utils.data.DataLoader(my_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=pts_collate_seg)


net=pointnet2_seg(input_dim=3)
if is_GPU:
    net=net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
critenrion=nn.NLLLoss()



def save_checkpoint(epoch,model,num_iter):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'iter':num_iter,
    },resume)

def log(filename,epoch,batch,loss,acc):
    f1=open(filename,'a')
    if epoch == 0 and batch == 0:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    f1.write('\nin epoch{} batch{} loss={} acc={}'.format(epoch,batch,loss,acc))
    f1.close()


def evaluate(model_test):
    NUM_CLASSES = 16

    part_label = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]

    num_dataset_cls = np.array([341, 14, 11, 158, 704, 14, 159, 80, 286, 83, 51, 38, 44, 12, 31, 848])
    weight_cls = num_dataset_cls.astype(np.float32) * 1.0 / num_dataset_cls.sum().astype(np.float32)

    model_test.eval()
    total_correct = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_mIOU_class = [0 for _ in range(NUM_CLASSES)]

    data_eval = shapenet_dataset(datalist_path=args.data_eval)
    eval_loader = torch.utils.data.DataLoader(data_eval,num_workers=4,
                                  batch_size=4, shuffle=True, collate_fn=pts_collate_seg)
    print("dataset size:", len(eval_loader.dataset))

    for batch_idx, (pts, label, seg) in enumerate(eval_loader):
        ## pts [N,P,3] label [N,] seg [N,P]
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
            seg_label = Variable(seg.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
            seg_label = Variable(seg)

        ## pred [N,50,P]  trans [N,64,64]
        pred = net(pts)

        _, pred_index = torch.max(pred, dim=1)  ##[N,P]
        num_correct = (pred_index.eq(seg_label)).data.cpu().sum()
        print('in batch{} acc={}'.format(batch_idx, num_correct.item() * 1.0 / (4 * 2048)))
        total_correct += num_correct.item()

        ################
        ## compute mIOU
        iou_batch = []
        for i in range(pred.size()[0]):  ## B
            iou_pc = []
            for part in part_label[label[i]]:  ## for each shape
                gt = (seg[i] == part)  ## gt of this part_idx
                predict = (pred_index[i] == part).cpu()

                intersection = (gt + predict) == 2
                union = (gt + predict) >= 1

                # print(intersection)
                # print(union)
                # assert False

                if union.sum() == 0:
                    iou_part = 1.0
                else:
                    iou_part = intersection.int().sum().item() / (union.int().sum().item() + 0.0001)

                iou_pc.append(iou_part)
                ##np.asarray(iou_pc).mean()  the mIOU of this shape

            #iou_batch.append(np.asarray(iou_pc).mean())
            total_mIOU_class[label[i]] += np.asarray(iou_pc).mean()
            total_seen_class[label[i]] += 1
    ## mIOU of each class
    mIOU_class = np.array(total_mIOU_class) / np.array(total_seen_class,dtype=np.float)

    with open(logname,'w+') as f:
        f.write('##############################################################')
        f.write('the average correct rate:{}'.format(total_correct * 1.0 / (len(eval_loader.dataset) * 2048)))
        f.write('the mean IOU overall :{}'.format((mIOU_class * weight_cls).sum()))
        f.write('##############################################################')


def train():

    net.train()
    num_iter=0
    start_epoch=0

    if os.path.exists(resume):
        if is_GPU:
            checkoint = torch.load(resume)
        else:
            checkoint = torch.load(resume, map_location=lambda storage, loc: storage)
        start_epoch = checkoint['epoch']
        net.load = net.load_state_dict(checkoint['model'])
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("Warining! No resume checkpoint to load")

    print('start training')

    for epoch in xrange(start_epoch,args.epochs):
        init_epochtime = time.time()

        for batch_idx, (pts, label, seg) in enumerate(data_loader):
            ## pts [N,P,3] label [N,] seg [N,P]
            t1=time.time()
            if is_GPU:
                pts = Variable(pts.cuda())
                label = Variable(label.cuda())
                seg_label = Variable(seg.cuda())
            else:
                pts = Variable(pts)
                label = Variable(label)
                seg_label = Variable(seg)

            ## pred [N,50,P]
            pred = net(pts)

            loss = critenrion(pred, seg_label)

            _, pred_index = torch.max(pred, dim=1) ##[N,P]

            num_correct = (pred_index.eq(seg_label)).data.cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2=time.time()
            num_iter+=1

            if num_iter%1==0:
                print('In Epoch{} Iter{},loss={} accuracy={}  time cost:{}'.format(epoch, num_iter, loss.data,
                                                                                   num_correct.item() / (args.batch_size*2048),
                                                                                   t2 - t1))
            if num_iter%(args.log_step*10)==0 and num_iter!=0:
                save_checkpoint(epoch, net, num_iter,)
                evaluate(net)
            if num_iter%(args.log_step)==0 and num_iter!=0:
                log(logname, epoch, num_iter, loss.data,num_correct.item() / (args.batch_size*2048))

            if (num_iter*args.batch_size)%args.decay_step==0 and num_iter!=0:
                f1 = open(logname, 'a')
                f1.write("learning rate decay in iter{}\n".format(num_iter))
                f1.close()
                print ("learning rate decay in iter{}\n".format(num_iter))
                for param in optimizer.param_groups:
                    param['lr'] *= args.decay_rate
                    param['lr'] = max(param['lr'],1e-5)

        end_epochtime = time.time()
        print('--------------------------------------------------------')
        print('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print('-------------------------------------------------------- \n')

    save_checkpoint(args.epochs-1, net, num_iter)
    evaluate(net)

if __name__ == '__main__':
    train()
