import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.pointnet2_module import SA_module,FP_module

class pointnet2_semseg(nn.Module):
    def __init__(self,input_dim,num_class=13,use_FPS=False):
        super(pointnet2_semseg, self).__init__()
        self.num_class=num_class

        ## 4096
        self.SA1=SA_module(num_sample=1024,num_nn=32,mlp_list=[32,32,64],input_dim=(input_dim+3),use_FPS=use_FPS)
        ## 1024
        self.SA2=SA_module(num_sample=256,num_nn=32,mlp_list=[64,64,128],input_dim=(64+3),use_FPS=use_FPS)
        ## 256
        self.SA3=SA_module(num_sample=64,num_nn=32,mlp_list=[128,128,256],input_dim=(128+3),use_FPS=use_FPS)
        ## 64
        self.SA4=SA_module(num_sample=16,num_nn=32,mlp_list=[256,256,512],input_dim=(256+3),use_FPS=use_FPS)
        ## 16

        self.FP1 = FP_module(input_dim=(512+256),mlp_list=[256,256])
        self.FP2 = FP_module(input_dim=(256+128),mlp_list=[256,256])
        self.FP3 = FP_module(input_dim=(256+64),mlp_list=[256,128])
        self.FP4 = FP_module(input_dim=(128+input_dim),mlp_list=[128,128,128])

        self.classifer=nn.Sequential(
            nn.Conv1d(128,128,1,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, self.num_class,1,1)
        )
    def forward(self,pts):
        ## pts [bs,P1,C1]
        pts_xyz = pts[:, :, :3]

        l1_indices, l1_xyz, l1_feature = self.SA1(pts_xyz, pts)  ##(bs, 1024, 32), (bs, 1024, 3), (bs, 1024, 64)
        l2_indices, l2_xyz, l2_feature = self.SA2(l1_xyz, l1_feature) ##(bs, 256, 32), (bs, 256, 3), (bs, 256, 128)
        l3_indices, l3_xyz, l3_feature = self.SA3(l2_xyz, l2_feature)
        l4_indices, l4_xyz, l4_feature = self.SA4(l3_xyz, l3_feature)

        #print (l2_xyz.size(), l3_xyz.size(), l2_feature.size(), l3_feature.size())
        new_l3_feature = self.FP1(l3_xyz, l4_xyz, l3_feature, l4_feature)
        new_l2_feature = self.FP2(l2_xyz, l3_xyz, l2_feature, new_l3_feature)
        new_l1_feature = self.FP3(l1_xyz, l2_xyz, l1_feature, new_l2_feature)
        new_pts_feature = self.FP4(pts_xyz, l1_xyz, pts, new_l1_feature)  ##[bs, P0, C]
        #print (new_pts_feature.size())
        new_pts_feature = new_pts_feature.transpose(1, 2)

        scores = self.classifer(new_pts_feature)
        prob = F.log_softmax(scores, dim=1) ##[bs, 50, P0]

        return prob


if __name__ == '__main__':
    net = pointnet2_semseg(input_dim=6)
    is_GPU = torch.cuda.is_available()
    if is_GPU:
        net = net.cuda()

    N = 3
    P1 = 4096
    pts_feature = torch.randn(N, P1, 6)
    if is_GPU:
        pts_feature = pts_feature.cuda()

    prob = net(pts_feature)
    print (prob.size())
