import os
import cv2
import numpy as np
import pandas as pd
from imageio import imread
from tqdm import tqdm
import matplotlib.pyplot as plt

def img_predict(matrix1):
    batch_idx = 3  # 选择第一个batch
    fig, axes = plt.subplots(1, 5, figsize=(40, 8))  # 一行十列的画布，每张图大小为40x8
    for channel_idx in range(5):
        image1 = matrix1[batch_idx, channel_idx, :, :]
        ax1 = axes[channel_idx]
        ax1.imshow(image1, cmap='gray')  # 假设是灰度图像，使用'gray'颜色映射
        ax1.set_title(f"Matrix 1 - Channel {channel_idx}")
        ax1.axis('off')  # 不显示坐标轴
    plt.show()

def gen_raw_data(train_frame_nums, img_path, img_size, save_path,channel):
    size = img_size
    position = int(train_frame_nums/mig_num)
    data = np.ndarray(shape=(position, mig_num, 256, 256), dtype=np.float32)
    folder_list = os.listdir(img_path)
    shape1 = 0
    for folders in tqdm(folder_list[:position]):
        img_list = os.listdir(img_path + folders)
        # img_list.insert(0, 'radar_001.png')
        shape2 = 0
        for img in img_list:
            imgPath = os.path.join(img_path + folders,img)
            img = np.array(imread(imgPath) / 255)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = np.float32(img)

            # imgb = np.where(img >= 10, 0, img / 70)
            # data[shape1, shape2, 0, :, :] = imgb
            #
            # imgb = np.where(img >= 20, 0, img)
            # imgb = np.where(imgb < 10, 0, imgb / 70)
            # data[shape1, shape2, 1, :, :] = imgb
            #
            # imgb = np.where(img >= 30, 0, img)
            # imgb = np.where(imgb < 20, 0, imgb / 70)
            # data[shape1, shape2, 2, :, :] = imgb
            #
            # imgb = np.where(img >= 40, 0, img)
            # imgb = np.where(imgb < 30, 0, imgb / 70)
            # data[shape1, shape2, 3, :, :] = imgb
            #
            # imgb = np.where(img >= 70, 0, img)
            # imgb = np.where(imgb < 40, 0, imgb / 70)
            # data[shape1, shape2, 4, :, :] = imgb

            data[shape1, shape2, :, :] = img
            shape2 += 1
        shape1 += 1
    savepath = save_path + 'valid_radar_256.npy'

    np.save(savepath, data)
    img_predict(data)
    print(savepath,data.shape)

def countFile(dir):
    filelist = os.listdir(dir)
    return len(filelist)



if __name__ == '__main__':

    npy_save_path = '/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/user_data/data/'
    train_dir = '/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/data/TestD/Radar/'
    frame_size = 128
    mig_num = 20
    channel = 5
    train_frame_nums = 1500 * mig_num
    val_frame_nums = countFile(train_dir) * mig_num
    ratio = []
    countFile(train_dir)
    gen_raw_data(val_frame_nums,img_path=train_dir, img_size=frame_size, save_path=npy_save_path,channel = channel)




