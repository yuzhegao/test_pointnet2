import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# from utils.sample_utils import furthest_point_sample


def gather_2d(xyz, index):
    """
    ues 2d-indices to gather value in 3d-tensor
    :param xyz: [B,N1,C]
    :param index: [B,N2]
    :return: [B,N2,C]
    """
    ndim = xyz.size(2)
    index = torch.unsqueeze(index, dim=2).repeat(1, 1, ndim)
    new_xyz = torch.gather(xyz, dim=1, index=index)

    return new_xyz


def random_sample(xyz, num_sample):
    """
     xyz: [bs,P1,3]
    :return:
     idx_sample: [bs,P2]
     new xyz:[bs,P2,3]
    """
    bs, P1, _ = xyz.size()
    idx = []
    for i in range(bs):
        a = torch.randperm(P1)[:num_sample]
        idx.append(a)
    idx_sample = torch.stack(idx, dim=0)
    new_xyz = gather_2d(xyz, idx_sample)

    return idx_sample, new_xyz


def get_distance_matrix(A, B):
    r_A = torch.unsqueeze(torch.sum(A * A, dim=2), dim=2)  ##[N,Pa,1]
    r_B = torch.unsqueeze(torch.sum(B * B, dim=2), dim=2)  ##[N,Pb,1]
    m = torch.bmm(A, torch.transpose(B, 2, 1))  ##[N,Pa,Pb]
    D = r_A - 2 * m + torch.transpose(r_B, 2, 1)

    return D  ##[N,Pa,Pb]


def KNN(new_xyz, input_xyz, num_nn):
    """
    get KNN pts for each key pts and normalize the coordinate
     new_xyz:  [bs,P2,3]
     input_xyz: [bs,P1,3]
     num_nn: int
    :return:
     nn_indices: [bs,P2,num_nn]
     knn_xyz: [bs,P2,num_nn,3]
    """
    bs, P1, _ = input_xyz.size()
    P2 = new_xyz.size(1)

    D = get_distance_matrix(new_xyz, input_xyz)  ##[bs,P2,P1]

    nn_distances, nn_indices = torch.topk(-D, k=num_nn)  ##[bs,P2,num_nn]

    nn_indices_patten = nn_indices.view(bs, P2 * num_nn)  ##[bs,P2*num_nn]
    knn_xyz = gather_2d(input_xyz, nn_indices_patten)  ##[bs,P2*num_nn,3]
    knn_xyz = knn_xyz.view(bs, P2, num_nn, 3)  ##[bs,P2,num_nn,3]
    knn_xyz -= torch.unsqueeze(new_xyz, dim=2).repeat(1, 1, num_nn, 1)

    return nn_indices, knn_xyz


##################################################################################

class SA_module(nn.Module):
    def __init__(self, num_sample, num_nn, mlp_list, input_dim, grouping_all=False):
        super(SA_module, self).__init__()
        self.num_sample = num_sample
        self.num_nn = num_nn
        self.grouping_all = grouping_all
        self.mlp = mlp_list

        mlp1, mlp2, mlp3 = mlp_list
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, mlp1, 1, 1),
            nn.BatchNorm1d(mlp1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mlp1, mlp2, 1, 1),
            nn.BatchNorm1d(mlp2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(mlp2, mlp3, 1, 1),
            nn.BatchNorm1d(mlp3),
            nn.ReLU(),
        )
        self.maxpool = torch.nn.MaxPool1d(self.num_nn)

    def forward(self, input_xyz, input_feature):
        """
         input_xyz: [bs,P1,3]
         input_feat: [bs,P1,C1]
        :return:
         new_xyz:[bs,P2,3]
         new_feat: [bs,P2,C1]
         idx: [bs,P1,num_nn]
        """
        batch_size, P1, C1 = input_feature.size()

        ###########################################
        # sampling
        idx, new_xyz = random_sample(input_xyz, self.num_sample)

        # idx=furthest_point_sample(input_xyz,self.num_sample)
        # new_xyz=gather_2d(input_xyz,idx)

        ###########################################
        # grouping
        nn_indices, grouped_xyz = KNN(new_xyz, input_xyz, self.num_nn)
        ##[bs,P2,num_nn] [bs,P2,num_nn,3]

        nn_indices_patten = nn_indices.view(batch_size, self.num_sample * self.num_nn)
        grouped_feature = gather_2d(input_feature, nn_indices_patten)
        grouped_feature = grouped_feature.contiguous().view(batch_size, self.num_sample, self.num_nn, C1)
        ##[bs,P2,num_nn,C1]
        grouped_feature = torch.cat([grouped_feature,grouped_xyz],dim=3)

        grouped_feature = grouped_feature.view(batch_size, self.num_sample * self.num_nn, -1).transpose(2,1)  ##[bs, C1+3, P2*num_nn]
        knn_feature = self.conv3(self.conv2(self.conv1(grouped_feature)))  ##[bs, C, P2*num_nn]

        knn_feature = knn_feature.transpose(2, 1)
        knn_feature = knn_feature.contiguous().view(batch_size, self.num_sample, self.num_nn, -1)  ##[bs, P2, num_nn, C]

        ##########################################
        ## pooling
        knn_feature = knn_feature.contiguous().view(batch_size * self.num_sample, self.num_nn, -1).transpose(2, 1)
        ##[bs*P2, C, num_nn]
        new_feature = F.max_pool1d(knn_feature, kernel_size=self.num_nn)
        new_feature = torch.squeeze(new_feature).contiguous().view(batch_size, self.num_sample, -1)

        return new_xyz, new_feature


if __name__ == '__main__':
    N = 9
    P1 = 200
    P2 = 50
    num_nn = 5

    pts_feature = torch.randn(N, P1, 6)
    pts = pts_feature[:,:,:3]

    SA1 = SA_module(num_sample=P2, num_nn=num_nn, mlp_list=[8, 32, 64], input_dim=(3+6))
    _, new_feature = SA1(pts, pts_feature)
    print (new_feature.size())