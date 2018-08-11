import os, sys
sys.path.append("../utils/")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


is_GPU=torch.cuda.is_available()

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
    if is_GPU:
        idx_sample = idx_sample.cuda()
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
     nn_distances: [bs,P2,num_nn]
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

    return nn_distances,nn_indices, knn_xyz


##################################################################################

class SA_module(nn.Module):
    def __init__(self, num_sample, num_nn, mlp_list, input_dim, grouping_all=False,use_FPS=False):
        super(SA_module, self).__init__()
        self.num_sample = num_sample
        self.num_nn = num_nn
        self.grouping_all = grouping_all
        self.fps=use_FPS
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

        if not self.grouping_all:
            ###########################################
            # sampling
            if self.fps:
                from sample_utils import furthest_point_sample
                idx=furthest_point_sample(input_xyz.contiguous(),self.num_sample)
                new_xyz=gather_2d(input_xyz,idx.long())
            else:
                idx, new_xyz = random_sample(input_xyz, self.num_sample)

            ###########################################
            # grouping
            _,nn_indices, grouped_xyz = KNN(new_xyz, input_xyz, self.num_nn)
            ##[bs,P2,num_nn] [bs,P2,num_nn,3]

            nn_indices_patten = nn_indices.view(batch_size, self.num_sample * self.num_nn)
            grouped_feature = gather_2d(input_feature, nn_indices_patten)
            grouped_feature = grouped_feature.contiguous().view(batch_size, self.num_sample, self.num_nn, C1)
            ##[bs,P2,num_nn,C1]
            grouped_feature = torch.cat([grouped_feature, grouped_xyz], dim=3)

            grouped_feature = grouped_feature.view(batch_size,
                            self.num_sample * self.num_nn, -1).transpose(2,1)  ##[bs, C1+3, P2*num_nn]
            knn_feature = self.conv3(self.conv2(self.conv1(grouped_feature)))  ##[bs, C, P2*num_nn]

            knn_feature = knn_feature.transpose(2, 1)
            knn_feature = knn_feature.contiguous().view(batch_size, self.num_sample, self.num_nn,-1)  ##[bs, P2, num_nn, C]

        else:
            ## all pts grouping to origin, now  P2=1 num_nn=P1
            new_xyz = torch.zeros([batch_size,1,3])

            nn_indices = torch.from_numpy(np.tile(np.array(range(P1)).reshape((1, 1, P1)), (batch_size, 1, 1)))
            grouped_xyz = input_xyz.contiguous().view(batch_size, 1, P1, 3)
            ## [bs,1,P1] [bs,1,P1,3]

            grouped_feature = input_feature.contiguous().view(batch_size, 1, P1, -1) ## [bs,1,P1,C1]

            if is_GPU:
                new_xyz = new_xyz.cuda()
                nn_indices = nn_indices.cuda()

            grouped_feature = torch.cat([grouped_feature, grouped_xyz], dim=3) ##[bs,1,P1,C1+3]
            grouped_feature = grouped_feature.view(batch_size,
                                                P1, -1).transpose(2,1)  ##[bs, C1+3, P1*1]
            knn_feature = self.conv3(self.conv2(self.conv1(grouped_feature)))  ##[bs, C, P1*1]
            knn_feature = knn_feature.transpose(2, 1)
            knn_feature = knn_feature.contiguous().view(batch_size, 1, P1,-1)  ##[bs, 1, P1, C]


        ##########################################
        ## pooling
        knn_feature = knn_feature.contiguous().view(batch_size * self.num_sample, self.num_nn, -1).transpose(2, 1)
        ##[bs*P2, C, num_nn]
        if self.grouping_all:
            new_feature = F.max_pool1d(knn_feature, kernel_size=P1)
        else:
            new_feature = F.max_pool1d(knn_feature, kernel_size=self.num_nn)
        new_feature = torch.squeeze(new_feature).contiguous().view(batch_size, self.num_sample, -1)

        return nn_indices,new_xyz, new_feature


######################################################################
class FP_module(nn.Module):
    def __init__(self, mlp_list, input_dim,fp_nn=3):
        super(FP_module, self).__init__()
        self.fp_nn=fp_nn

        num_layer = len(mlp_list)
        if num_layer == 2:
            mlp1, mlp2 = mlp_list
            mlp3 = None
        else:
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
        self.conv3 = None
        if mlp3 is not None:
            self.conv3 = nn.Sequential(
                nn.Conv1d(mlp2, mlp3, 1, 1),
                nn.BatchNorm1d(mlp3),
                nn.ReLU(),
            )
    def forward(self,xyz1,xyz2,pts1,pts2):
        """
         xyz1: [N,P1,3] -> pts of previous layer
         xyz2: [N,P2,3] -> pts of late layer, here P1 > P2
         pts1: [N.P1,C1]
         pts2: [N,P2,C2]
        :return:
         new_points: (N, P1, mlp[-1]) -> P1 pts with new feature
        """
        batchsize, P1, C1 = pts1.size()
        _,P2,C2 = pts2.size()

        fp_disc, fp_indice, fp_xyz = KNN(xyz1, xyz2, num_nn=self.fp_nn)
        ##[bs,P1,3] [bs,P1,3] [bs,P1,3,3]
        fp_disc = torch.clamp(fp_disc, min=1e-10,max=10000)
        norm = torch.unsqueeze(torch.sum(1.0 / fp_disc, dim=2), dim=2).repeat(1,1,self.fp_nn) ##[bs,P1,3]
        weight = (1.0/fp_disc) / norm ##[bs,P1,3]
        weight = torch.unsqueeze(weight, dim=3).repeat(1,1,1,C2)

        ## feature propagation: [bs,P1,nn] + [bs,P2,C]2 -> [bs,P1,nn,C2]
        fp_indice_platten = fp_indice.contiguous().view(batchsize, P1 * self.fp_nn) ##[bs,P1*nn]
        fp_row_feature = gather_2d(pts2,fp_indice_platten)
        fp_row_feature = fp_row_feature.contiguous().view(batchsize, P1, self.fp_nn, C2) ##[bs,P1,3,C2]
        fp_weight_feature = fp_row_feature * weight ##[bs,P1,3,C1]

        fp_weight_feature = fp_weight_feature.contiguous().view(batchsize * P1,self.fp_nn, -1).transpose(2,1) ##[bs*P1,C2,3]
        fp_group_feature = torch.squeeze(F.max_pool1d(fp_weight_feature,kernel_size=self.fp_nn)) ##[bs*P1,C2]
        fp_group_feature = fp_group_feature.contiguous().view(batchsize, P1, C2) ## [bs,P1,C2]

        ## unet archtecture
        fp_group_feature = torch.cat([fp_group_feature,pts1], dim=2) ## [bs,P1,C2+C1]
        fp_group_feature = fp_group_feature.transpose(1,2) ## [bs,C2+C1,P1]

        new_feature = self.conv2(self.conv1(fp_group_feature))
        if self.conv3 is not None:
            new_feature = self.conv3(new_feature)
        new_feature = new_feature.transpose(2,1) ## [bs,P1,mlp[-1]]

        return new_feature







if __name__ == '__main__':
    N = 9
    P1 = 200
    P2 = 50
    num_nn = 5

    pts_feature = torch.randn(N, P1, 6)
    if is_GPU:
        pts_feature=pts_feature.cuda()
    pts = pts_feature[:,:,:3]

    ## test SA module
    SA1 = SA_module(num_sample=P2, num_nn=num_nn, mlp_list=[8, 32, 64],
                    input_dim=(3+6),use_FPS=False)
    if is_GPU:
        SA1 = SA1.cuda()
    _,_, new_feature = SA1(pts, pts_feature)
    print (new_feature.size())

    ## test SA grouping all
    SA2=SA_module(num_sample=1, num_nn=P1, mlp_list=[8, 32, 64],
                  input_dim=(3+6),use_FPS=False,grouping_all=True)
    if is_GPU:
        SA2 = SA2.cuda()
    _,_, new_feature = SA2(pts, pts_feature)
    print (new_feature.size())
    print ('\n')
    ###############################################################################

    xyz1 = torch.randn(N, P1, 3)
    xyz2 = torch.randn(N, P2, 3)
    pts1 = torch.randn(N, P1, 6)
    pts2 = torch.randn(N, P2, 19)
    if is_GPU:
        xyz1,xyz2,pts1,pts2 = xyz1.cuda(),xyz2.cuda(),pts1.cuda(),pts2.cuda()

    ## test FP module
    FP1 = FP_module(input_dim=(6+19),mlp_list=[8,32,64])
    if is_GPU:
        FP1 = FP1.cuda()
    new_feat=FP1(xyz1,xyz2,pts1,pts2)
    print (new_feat.size())


