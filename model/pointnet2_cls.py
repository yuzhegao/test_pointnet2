import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2_module import SA_module


class pointnet2_cls(nn.Module):
    def __init__(self,input_dim,num_class=40,use_FPS=False):
        super(pointnet2_cls, self).__init__()
        self.num_class=num_class

        self.SA1=SA_module(num_sample=512,num_nn=32,mlp_list=[64,64,128],input_dim=(input_dim+3),use_FPS=use_FPS)
        self.SA2=SA_module(num_sample=128,num_nn=64,mlp_list=[128,128,256],input_dim=(128+3),use_FPS=use_FPS)
        self.SA3=SA_module(num_sample=1,num_nn=128,mlp_list=[256,512,1024],input_dim=(256+3),use_FPS=use_FPS,grouping_all=True)

        self.classifer=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(256, self.num_class),
        )
    def forward(self,pts):
        ## pts [bs,P1,C1]
        pts_xyz = pts[:, :, :3]

        l1_indices, l1_xyz, l1_feature = self.SA1(pts_xyz, pts)  ##(bs, 512, 32), (bs, 512, 3), (bs, 512, 128)
        l2_indices, l2_xyz, l2_feature = self.SA2(l1_xyz, l1_feature)  ##(bs, 128, 64), (bs, 128, 3), (bs, 128, 256)
        l3_indices, l3_xyz, l3_feature = self.SA3(l2_xyz, l2_feature)  ##(bs, 1, 128), (bs, 1, 3), (bs, 1, 1024)

        global_feature = torch.squeeze(l3_feature)
        scores = self.classifer(global_feature) ##[bs,40]
        prob = F.log_softmax(scores, dim=-1)

        return prob

if __name__ == '__main__':
    net = pointnet2_cls(input_dim=6)
    is_GPU = torch.cuda.is_available()
    if is_GPU:
        net = net.cuda()

    N = 2
    P1 = 1024
    pts_feature = torch.randn(N, P1, 6)
    if is_GPU:
        pts_feature = pts_feature.cuda()

    prob = net(pts_feature)
    print (prob.size())