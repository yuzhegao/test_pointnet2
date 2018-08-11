import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2_module import SA_module,FP_module

class pointnet2_seg(nn.Module):
    def __init__(self,input_dim,num_class=50,use_FPS=False):
        super(pointnet2_seg, self).__init__()
        self.num_class=num_class

        ## 1024
        self.SA1=SA_module(num_sample=512,num_nn=32,mlp_list=[64,64,128],input_dim=(input_dim+3),use_FPS=use_FPS)
        ## 512
        self.SA2=SA_module(num_sample=128,num_nn=64,mlp_list=[128,128,256],input_dim=(128+3),use_FPS=use_FPS)
        ## 128
        self.SA3=SA_module(num_sample=1,num_nn=128,mlp_list=[256,512,1024],input_dim=(256+3),use_FPS=use_FPS,grouping_all=True)
        ## 1

        self.FP1 = FP_module(input_dim=(1024+256),mlp_list=[256,256],fp_nn=1)
        self.FP2 = FP_module(input_dim=(256+128),mlp_list=[256,128])
        self.FP3 = FP_module(input_dim=(128+input_dim),mlp_list=[128,128,128])

        self.classifer=nn.Sequential(
            nn.Conv1d(128,128,1,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Conv1d(128, self.num_class,1,1)
        )
    def forward(self,pts):
        ## pts [bs,P1,C1]
        pts_xyz = pts[:, :, :3]

        l1_indices, l1_xyz, l1_feature = self.SA1(pts_xyz, pts)  ##(bs, 512, 32), (bs, 512, 3), (bs, 512, 128)
        l2_indices, l2_xyz, l2_feature = self.SA2(l1_xyz, l1_feature)  ##(bs, 128, 64), (bs, 128, 3), (bs, 128, 256)
        l3_indices, l3_xyz, l3_feature = self.SA3(l2_xyz, l2_feature)  ##(bs, 1, 128), (bs, 1, 3), (bs, 1, 1024)

        print (l2_xyz.size(), l3_xyz.size(), l2_feature.size(), l3_feature.size())
        new_l2_feature = self.FP1(l2_xyz, l3_xyz, l2_feature, l3_feature)
        new_l1_feature = self.FP2(l1_xyz, l2_xyz, l1_feature, new_l2_feature)
        new_pts_feature = self.FP3(pts_xyz, l1_xyz, pts, new_l1_feature)  ##[bs, P0, C]
        print (new_pts_feature.size())
        new_pts_feature = new_pts_feature.transpose(1, 2)

        scores = self.classifer(new_pts_feature)
        prob = F.log_softmax(scores, dim=1) ##[bs, 50, P0]

        return prob


if __name__ == '__main__':
    net = pointnet2_seg(input_dim=6)
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