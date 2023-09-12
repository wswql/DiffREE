import cv2
import numpy as np
import torch
import pandas as pd
import  os
from torchvision import datasets, transforms
from numpy import random

tablepath = '../data/radar/train/table.csv'

train_csv = pd.read_csv(tablepath)
indexlist = train_csv.values[:, 0]


len = len(indexlist)
data = np.ndarray(shape=(len, 48, 2,200,200 ), dtype=np.float32)

for ind in range(len):
    index = indexlist[ind]
    for j,i in enumerate(range(int(index.split('.')[0].split('_')[-1]), int(index.split('.')[0].split('_')[-1])+48)):
        imagepath = os.path.join('/data03/wuqiliang/code/mcvd_Sample/data/radar/train/train',str(index.split('.')[0].split('_')[1])+'/'+"2d_"+ index.split('.')[0].split('_')[1]+"_"+str(i)+".png")
        img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)/250)
        img = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
        data[ind][j][0] = img[:, :200]
        data[ind][j][1] = img[:, 200:]
print(data.shape)





#
#
# def load_fixed_set(root, filename):
#     path = os.path.join(root, filename)
#     dataset = np.load(path)
#     return dataset
#
# class MovingRadar_V4(object):
#     def __init__(self,seq_len=20, is_train=True):
#         self.tablepath = '../data/radar/train/table.csv'
#         self.tablepath = '../data/radar/train/table.csv'
#         self.seq_len = seq_len
#         self.train_csv = pd.read_csv(self.tablepath)
#         self.indexlist = self.train_csv.values[:, 0]
#         self.length = len(self.indexlist)
#         self.data = np.ndarray(shape=(48, 2, 200, 200), dtype=np.float32)
#         self.label = ['ob', '850t', 'cape', '850u', '850v', '500u', '500v', 'dbz', '850rh', 'prmsl', '500t', '500rh', '10v', '2d', '10u', '2t']
#         self.random = random.randint(16)
#         if is_train:
#             self.indexlist =  self.indexlist[:13000]
#             self.length = len(self.indexlist)
#         else:
#             self.indexlist =  self.indexlist[13000:]
#             self.length = len(self.indexlist)
#
#
#
#
#     def __getitem__(self, idx):
#         index = self.indexlist[idx]
#         start = int(index.split('.')[0].split('_')[-1])
#         end = int(index.split('.')[0].split('_')[-1]) + 48
#         for j, i in enumerate(range(start, end)):
#             labelimg = str(index.split('.')[0].split('_')[1]) + '/' + self.label[self.random]+'_' +index.split('.')[0].split('_')[1] + "_" + str(i) + ".png"
#             imagepath = os.path.join('/data03/wuqiliang/code/mcvd_Sample/data/radar/train/train',labelimg)
#             img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)/250)
#             img = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
#             self.data[j][0] = img[:, :200]
#             self.data[j][1] = img[:, 200:]
#         randomnum = random.randint(4)
#         if randomnum == 0:
#             images = self.data[0:self.seq_len, ...]
#         elif randomnum == 1:
#             images = self.data[1:self.seq_len+1, ...]
#         elif randomnum == 2:
#             images = self.data[2:self.seq_len+2, ...]
#         elif randomnum == 3:
#             images = self.data[3:self.seq_len+3, ...]
#         temp = np.zeros_like(images)
#         return torch.tensor(images), temp
#     def __len__(self):
#         return self.length
#
#
# if __name__ == '__main__':
#
#     nn = MovingRadar_V4(seq_len=18,is_train=False)
#
#     outs = nn.__getitem__(1888)
#     print(outs[0].shape, nn.__len__())