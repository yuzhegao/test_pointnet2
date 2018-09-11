from __future__ import print_function,division

import os
import h5py
import time
import numpy as np

import torch
import torchvision
import torch.utils.data as data

## use to load point clouds and class_label from .h5
def load_cls(filelist,use_color=False):
    ## here data ->[num_pts,9]
    points = []
    labels = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        filename = os.path.basename(line.rstrip())
        f = h5py.File(os.path.join(folder, filename))
        data, label = f['data'][...], f['data'][...]
        if use_color:
            data = data[:,:6]
        else:
            data = data[:,:3]

        points.append(data.astype(np.float32))
        labels.append(np.squeeze(label).astype(np.int64))
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


###########################
# indoor 3d
# indoor3d_sem_seg_hdf5_data.zip
###########################
class indoor3d_dataset(data.Dataset):
    def __init__(self, datalist_path, use_color=False, test_area='Area_6', training=True):
        ## datalist_path: .txt file of hdf5 dataset
        ## if test_area is area's name, then for a whole area inference; if is room's name, then can be used for single room evaluation

        self.datalist = datalist_path
        self.test_area = test_area

        self.pts,self.seg_label=load_cls(self.datalist, use_color=use_color)
        print ('Total: data {} label {}'.format(self.pts.shape,self.seg_label.shape))

        root= os.path.dirname(self.datalist)
        room_filelist = [line.rstrip() for line in open(os.path.join(root,'room_filelist.txt'))]

        data_idx=[]
        if training:
            for i, room_name in enumerate(room_filelist):
                if test_area not in room_name:
                    data_idx.append(i)
        else:
            for i, room_name in enumerate(room_filelist):
                if test_area in room_name:
                    data_idx.append(i)
        self.pts=self.pts[data_idx,...]
        self.seg_label=self.seg_label[data_idx,...]

        print ('data size:{} label size:{}'.format(self.pts.shape,self.seg_label.shape))


    def __getitem__(self, index):
        pts, seg_label = self.pts[index], self.seg_label[index]

        return pts, seg_label

    def __len__(self):
        return len(self.pts)

def pts_collate_seg(batch):
    pts_batch=[]
    seg_label_batch=[]

    for sample in batch:
        pts_batch.append(torch.from_numpy(sample[0]))
        seg_label_batch.append(torch.from_numpy(sample[1]))

    pts_batch=torch.stack(pts_batch,dim=0)  ##[bs,P,3]
    pts_batch=torch.transpose(pts_batch,dim0=1,dim1=2)
    seg_label_batch=torch.stack(seg_label_batch,dim=0)

    return pts_batch.float(),seg_label_batch.long()



if __name__ == '__main__':
    dataset=indoor3d_dataset(datalist_path='/../../3d_data/indoor3d_sem_seg_hdf5_data/all_files.txt', training=False, use_color=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=2, shuffle=True, collate_fn=pts_collate_seg)

    for idx,(pts,seg_label) in enumerate(loader):
        print (pts.size())
        print (seg_label.size())
        break


