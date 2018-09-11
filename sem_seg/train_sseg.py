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

parser.add_argument('--fps', action='store_true', help='Whether to use fps')
parser.add_argument('--color', action='store_true', help='Whether to use color')
parser.add_argument('--resume', default=None,type=str, metavar='PATH',help='path to latest checkpoint ')

args=parser.parse_args()

LOG_DIR = os.path.join(args.log,time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time())))
print ('prepare training in {}'.format(time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time()))))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if args.resume is None:
    resume = os.path.join(LOG_DIR, "checkpoint.pth")
else:
    resume = args.resume

logname = os.path.join(LOG_DIR,'log.txt')
optfile = os.path.join(LOG_DIR,'opt.txt')
with open(optfile, 'wt') as opt_f:
    opt_f.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        opt_f.write('%s: %s\n' % (str(k), str(v)))
    opt_f.write('-------------- End ----------------\n')

if is_GPU:
    torch.cuda.set_device(args.gpu)

my_dataset = indoor3d_dataset(args.data, test_area=args.area_eval)
data_loader=torch.utils.data.DataLoader(my_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=pts_collate_seg)

data_eval = indoor3d_dataset(args.data_eval, test_area=args.area_eval,training=False)
eval_loader = torch.utils.data.DataLoader(data_eval,num_workers=4,
            batch_size=4, shuffle=True, collate_fn=pts_collate_seg)

if args.color:
    feat_dim = 6
else:
    feat_dim = 3
net=pointnet2_seg(input_dim=feat_dim, use_FPS=args.fps)
if is_GPU:
    net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
critenrion = nn.NLLLoss()

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

        ## pred [N,50,P]  trans [N,64,64]
        pred = net(pts)

        _, pred_index = torch.max(pred, dim=1)  ##[N,P]
        pred_index, seg_label = pred_index.view(-1,), seg_label.view(-1,)
        num_correct = (pred_index.eq(seg_label)).data.cpu().sum()
        print('in batch{} acc={}'.format(batch_idx, num_correct.item() * 1.0 / (4 * 4096)))
        total_correct += num_correct.item()

        for idx,pts_label in enumerate(seg_label):
            total_seen_class[pts_label] += 1
            total_correct_class[pts_label] += (pred_index.eq(pts_label))[idx]

    with open(logname,'a') as f:
        f.write('\nthe accuracy:{}\n'.format(total_correct * 1.0 / (len(eval_loader.dataset) * 4096)))
        f.write('eval avg class acc: %f \n\n' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


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

        for batch_idx, (pts, seg) in enumerate(data_loader):
            ## pts [N,P,3] label [N,] seg [N,P]
            t1=time.time()
            if is_GPU:
                pts = Variable(pts.cuda())
                seg_label = Variable(seg.cuda())
            else:
                pts = Variable(pts)
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
                print ("\n\nlearning rate decay in iter{}\n".format(num_iter))
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
