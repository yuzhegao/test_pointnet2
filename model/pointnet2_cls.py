import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2_module import SA_module

"""
npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None
npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None
npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None
classifer FC 512 - 256 - 40
"""

class pointnet2_cls(nn.Module):
    def __init__(self,input_dim,num_class=40):
        super(pointnet2_cls, self).__init__()
        self.num_class=num_class

        self.SA1=SA_module(num_sample=512,num_nn=32,mlp_list=[64,64,128],input_dim=(input_dim+3))
        self.SA2=SA_module(num_sample=128,num_nn=64,mlp_list=[128,128,256],input_dim=(128+3))
        self.SA3=SA_module(num_sample=1,num_nn=128,mlp_list=[256,512,1024],input_dim=(256+3),grouping_all=True)

    def forward(self,pts):
        pass