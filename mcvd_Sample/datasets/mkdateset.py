import os

import cv2
import numpy as np
import pandas as pd
from imageio import imread
from tqdm import tqdm




def mk_train_dataset(npy_save_path,row,h,w):
    data = np.ndarray(shape=(row, h, w), dtype=np.float32)
    strnum = [str(0)*(5-len(str(i)))+str(i) for i in range(1,row+1)]
    img_row_path = [f'./data/Train/Radar/radar_{i}.png'for i in strnum]
    for i ,img_path in enumerate(tqdm(img_row_path)):
        img = np.array(imread(img_path) / 255)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = np.float32(img)
        data[i] = img
    train_savepath = npy_save_path + 'train_radar_input_data.npy'
    np.save(train_savepath, data)
    print(train_savepath)



def mk_test_dataset(npy_save_path,row,h,w):

    data = np.ndarray(shape=(row, h, w), dtype=np.float32)
    strnum = [str(0)*(5-len(str(i)))+str(i) for i in range(31218,34582+1)]
    img_row_path = [f'/data/TestD/Radar/radar_{i}.png'for i in strnum]
    for i ,img_path in enumerate(tqdm(img_row_path)):
        img = np.array(imread(img_path) / 255)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = np.float32(img)
        data[i] = img
    train_savepath = npy_save_path + 'test_radar_input_data.npy'
    np.save(train_savepath, data)
    print(train_savepath,data.shape)


if __name__ == '__main__':
    test = 0
    if test:
        print("-------------testdata------------------")
        npy_save_path = './user_data/data/'
        row = 3367
        h = int(480)
        w = int(560)
        mk_test_dataset(npy_save_path,row,h,w)
    else:
        print("-------------traindata------------------")
        npy_save_path = './user_data/data/'
        row = 31216
        mig_num = 41
        h = int(480)
        w = int(560)
        mk_train_dataset(npy_save_path,row,h,w)


