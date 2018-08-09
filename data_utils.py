from __future__ import print_function,division

import os
import h5py
import time
import numpy as np

import torch
import torchvision
import torch.utils.data as data

## use to load point clouds and class_label from .h5
def load_cls(filelist,use_extra_feature=False):
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))
        #print (data['data'][...].shape) ##(2048, 2048, 3)
        if 'normal' in data and use_extra_feature:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))

def load_seg(filelist):
    points = []
    labels = []
    seg_labels=[]

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        data = h5py.File(os.path.join(folder, filename))

        points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
        seg_labels.append(data['pid'][:].astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(seg_labels, axis=0))


#########################
# modelnet40
# modelnet40_ply_hdf5_2048.zip
#########################
class pts_cls_dataset(data.Dataset):
    def __init__(self,datalist_path,num_points=1024,data_argument=True,use_extra_feature=False):
        super(pts_cls_dataset, self).__init__()
        self.num_points=num_points
        self.data_argument=data_argument
        self.extra_feature=use_extra_feature
        self.pts, self.label = load_cls(datalist_path,use_extra_feature=use_extra_feature)
        print ('data size:{} label size:{}'.format(self.pts.shape,self.label.shape))

    def __getitem__(self, index):
        pts,label=self.pts[index],self.label[index]
        if len(pts) > self.num_points:
            choice=np.random.choice(len(pts),self.num_points,replace=False)
            pts=pts[choice]

        ## data argument
        if self.data_argument:
            """
            scale_param = np.random.uniform(low=0.66, high=1.5)
            pts[:, 2] = pts[:, 2] * scale_param
            pts[:, 0] = pts[:, 0] * scale_param  ## scale horizonal plane
            """

            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            pts = pts.dot(rotation_matrix)  ## rotate

            #pts = pts + np.random.uniform(low=-0.1, high=0.1, size=[1, 3])  ## translation [-0.1,0.1]

            sigma = 0.01
            clip = 0.05
            jittered_data = np.clip(sigma * np.random.randn(self.num_points, 3), -1 * clip, clip)
            pts += jittered_data  ## jitter

        return pts,label

    def __len__(self):
        return len(self.label)


def pts_collate(batch):
    pts_batch=[]
    label_batch=[]

    for sample in batch:
        pts_batch.append(torch.from_numpy(sample[0]))
        label_batch.append(sample[1])

    pts_batch=torch.stack(pts_batch,dim=0)
    pts_batch=torch.transpose(pts_batch,dim0=1,dim1=2)
    label_batch =torch.from_numpy(np.squeeze(label_batch))

    return pts_batch.float(),label_batch.long()


###########################
# shapenet_core
# shapenet_part_seg_hdf5_data.zip
###########################
class shapenet_dataset(data.Dataset):
    def __init__(self, datalist_path):
        self.datalist = datalist_path
        root = os.path.dirname(datalist_path)

        classname_file = os.path.join(root, 'all_object_categories.txt')
        with open(classname_file, 'r') as fin:
            lines = [line.rstrip() for line in fin.readlines()]
            self.classname = [line.split()[0] for line in lines]
        print (self.classname)

        self.pts,self.label,self.seg_label=load_seg(self.datalist)
        print ('data size:{} label size:{}'.format(self.pts.shape,self.seg_label.shape))


    def __getitem__(self, index):
        pts, label, seg_label = self.pts[index], self.label[index],self.seg_label[index]

        return pts, label, seg_label

    def __len__(self):
        return len(self.pts)

def pts_collate_seg(batch):
    pts_batch=[]
    label_batch=[]
    seg_label_batch=[]

    for sample in batch:
        pts_batch.append(torch.from_numpy(sample[0]))
        label_batch.append(sample[1])
        seg_label_batch.append(torch.from_numpy(sample[2]))

    pts_batch=torch.stack(pts_batch,dim=0)  ##[bs,P,3]
    pts_batch=torch.transpose(pts_batch,dim0=1,dim1=2)
    label_batch =torch.from_numpy(np.squeeze(label_batch))
    seg_label_batch=torch.stack(seg_label_batch,dim=0)

    return pts_batch.float(),label_batch.long(),seg_label_batch.long()



if __name__ == '__main__':
    dataset=shapenet_dataset(datalist_path='/home/gaoyuzhe/Downloads/3d_data/hdf5_data/test_hdf5_file_list.txt')
    loader=torch.utils.data.DataLoader(dataset,batch_size=2, shuffle=True, collate_fn=pts_collate_seg)

    for idx,(pts,label,seg_label) in enumerate(loader):
        print (pts.size())
        print (seg_label.size())
        print (label.size())
        break


