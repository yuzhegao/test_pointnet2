import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function

from _ext import pointnet2

class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx,xyz,npoint):
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()

        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output
        )

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


if __name__ == '__main__':
    N=24
    P=1024
    init_xyz = torch.randn([N,P,3])
    new_xyz=furthest_point_sample(init_xyz,10)
    print (new_xyz.size())
    
    
