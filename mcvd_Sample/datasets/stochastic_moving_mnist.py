import glob
import time
import torch.utils.data as data
import numpy as np
import torch
import pandas as pd
import  os
from numpy import random
import cv2
from configs.tools import reshape_patch,reshape_patch_back
import imageio
import argparse
import yaml
from skimage.transform import resize
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_fixed_set(root, filename):
    path = os.path.join(root, filename)
    dataset = np.load(path)
    return dataset



class VaildSet(object):
    def __init__(self, config, root,train_path, n_frames_input=10):
        super(VaildSet, self).__init__()
        # self.rotate = RandomRotate([0, 90])
        self.config = config
        self.dataset = load_fixed_set(root,train_path)
        self.length = self.dataset.shape[0]
        self.n_frames_input = n_frames_input
    def __getitem__(self, idx):
        images = self.dataset[idx, :, ...]
        # images = self.rotate(images)
        images = images[5:self.n_frames_input+5]
        images = images.astype(float)
        images = np.nan_to_num(images, nan=0, copy=False)
        images[images < 0] = 0
        images = images[..., np.newaxis]
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2).to(torch.float32), temp
    def __len__(self):
        return self.length




class Weather(object):
    def __init__(self, config, root,):
        super(Weather, self).__init__()
        self.config = config
        self.dataset = glob.glob(root)
        self.length = len(self.dataset)
    def __getitem__(self, idx):
        path = self.dataset[idx]
        selected_indices = [65, 66, 67, 68, 69]
        data = torch.load(path)[:, selected_indices, ...]
        images = torch.reshape(data , (-1, 161, 161))
        target_size = (images.shape[0], 160, 160)
        resized_tensor = np.zeros(target_size)
        for i in range(target_size[0]):
            resized_tensor[i] = resize(images[i], target_size[1:], mode='constant')
        images = resized_tensor[..., np.newaxis]
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2).to(torch.float32), temp
    def __len__(self):
        return self.length



class VaildSet_v3(object):
    def __init__(self, config, root,train_path, n_frames_input=10):
        super(VaildSet_v3, self).__init__()
        # self.rotate = RandomRotate([0, 90])
        self.config = config
        self.dataset = load_fixed_set(root,train_path)
        self.index = load_fixed_set(root, 'Z9072_data_max70_index.npy')[-2000:]
        self.length = len(self.index)
        self.n_frames_input = n_frames_input
    def __getitem__(self, idx):
        num = self.index[idx]
        images = self.dataset[num:num+30, ...]
        # images = self.rotate(images)
        images = images.astype(float)
        images = np.nan_to_num(images, nan=0, copy=False)
        images[images < 0] = 0
        images = images[..., np.newaxis]
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2).to(torch.float32), temp
    def __len__(self):
        return self.length

class VaildSet_v2(object):
    def __init__(self, config, root,train_path, n_frames_input=10, lens=1460):
        super(VaildSet_v2, self).__init__()
        self.rotate = RandomRotate([0, 90])
        self.lens = lens
        self.config = config
        self.dataset = load_fixed_set(root,train_path)
        self.length = self.dataset.shape[0]
        self.n_frames_input = n_frames_input
    def __getitem__(self, idx):

        images = self.dataset[idx:idx+22, ...]
        images = np.reshape(images, (-1, images.shape[2], images.shape[3]))
        # images = self.rotate(images)
        # images = images[:self.n_frames_input]
        images = images[..., np.newaxis]
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2).to(torch.float32), temp
    def __len__(self):
        return self.lens - 21



class MovingRadar_V3(object):
    def __init__(self, config, seq_len=20, is_train=True, lens=2400):
        self.seq_len = seq_len
        self.config = config
        self.rotate = RandomRotate([0, 90])
        if is_train:
            self.dataset = np.load('/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/user_data/data/train_256_data.npy') * config.data.Weight
            self.train_csv = pd.read_csv('/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/data/Train.csv')
            self.start = self.train_csv.values[:, 0]

            # 可以手动调节训练集大小范围[0, 23793]
            self.length = 23000
        else:
            self.dataset = np.load('/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/user_data/data/train_256_data.npy') * config.data.Weight
            self.train_csv = pd.read_csv('/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/data/TestA.csv')
            self.start = self.train_csv.values[:, 0]
            # 可以手动调节测试集大小 范围[0, 2585]
            self.length = 500
        self.is_train = is_train

    def __getitem__(self, idx):
        if self.is_train:
            start = int(self.start[idx].split('.')[0]) - 1
            end = int(self.start[idx].split('.')[0]) + 40
            images = self.dataset[start:end]
        else:
            start = int(self.start[idx].split('.')[0]) - 1 -31217
            end = int(self.start[idx].split('.')[0]) + 40 - 31217
            images = self.dataset[start:end]

        images = self.rotate(images)
        images = images[:-1]
        images = np.transpose(images[..., np.newaxis],[0,3,1,2])
        randomnum = random.randint(4)
        if randomnum == 0:
            images = images[0:self.seq_len, ...]
        elif randomnum == 1:
            images = images[1:self.seq_len+1, ...]
        elif randomnum == 2:
            images = images[2:self.seq_len+2, ...]
        elif randomnum == 3:
            images = images[3:self.seq_len+3, ...]
        images = np.squeeze(images)
        images = np.expand_dims(images, -1)
        # Input Image Tensor Shape: torch.Size([2, 5, 256, 256, 1])
        # Output Patch Tensor Shape: torch.Size([2, 5, 128, 128, 4])
        images = reshape_patch(images, self.config.data.patch)

        if images.ndim == 3:
            images = images[..., np.newaxis]
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2), temp
    def __len__(self):
        return self.length


class MovingRadar_V4(object):
    def __init__(self, config, seq_len=5):
        self.seq_len = seq_len
        self.checkimg = np.array(cv2.imread('/data03/wuqiliang/code/mcvd_Sample/data/radar/ob_200015_12_b.png', cv2.IMREAD_UNCHANGED) / 250)
        self.config = config
        self.path = '/data03/wuqiliang/code/mcvd_Sample/data/radar/test_b'
        self.indexlist = [os.path.join(self.path , str(i)) for i in os.listdir(self.path )]
        self.length = len(self.indexlist)
        self.data = np.ndarray(shape=(12, 128, 256), dtype=np.float32)
    def __getitem__(self, idx):
        index = self.indexlist[idx]
        nums = index.split('/')[-1]
        for i in range(1, 13):
            filename = 'ob_' + nums + '_' + str(i) + '.png'
            imagepath = os.path.join(index, filename)
            # img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED) / 250) - self.checkimg
            img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED) / 250) - self.checkimg
            img = np.where(img < 0.01, 0, img)
            img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
            self.data[i-1] = img
        images = self.data[-self.seq_len:, ...]
        images = np.where(images > 0.9, 0, images)
        images = np.expand_dims(images, -1)
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2), temp
    def __len__(self):
        return self.length

class MovingRadar_V5(object):
    def __init__(self, config, seq_len=20,is_train=True):
        self.config = config
        self.checkimg = np.array(cv2.imread('/data03/wuqiliang/code/mcvd_Sample/data/radar/ob_200015_12.png', cv2.IMREAD_UNCHANGED) / 250)
        self.tablepath = '/data03/wuqiliang/code/mcvd_Sample/data/radar/train/table.csv'
        self.tablepath = '/data03/wuqiliang/code/mcvd_Sample/data/radar/train/table.csv'
        self.seq_len = seq_len
        self.train_csv = pd.read_csv(self.tablepath)
        self.indexlist = self.train_csv.values[:, 0]
        self.length = len(self.indexlist)
        # self.data = np.ndarray(shape=(48, 256, 512), dtype=np.float32)
        self.data = np.ndarray(shape=(48, 128, 256), dtype=np.float32)
        self.label = ['ob', 'dbz']
        self.random = random.randint(2)
        if is_train:
            self.indexlist = self.indexlist[:13000]
            self.length = len(self.indexlist)
        else:
            self.indexlist = self.indexlist[13000:]
            self.length = len(self.indexlist)
    def __getitem__(self, idx):
        index = self.indexlist[idx]
        start = int(index.split('.')[0].split('_')[-1])
        end = int(index.split('.')[0].split('_')[-1]) + 48
        for j, i in enumerate(range(start, end)):
            labelimg = str(index.split('.')[0].split('_')[1]) + '/' + self.label[0]+'_' +index.split('.')[0].split('_')[1] + "_" + str(i) + ".png"
            imagepath = os.path.join('/data03/wuqiliang/code/mcvd_Sample/data/radar/re_train/train',labelimg)
            img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED) / 250) - self.checkimg
            # img = np.array(cv2.imread(imagepath, cv2.IMREAD_UNCHANGED) / 250)
            img = np.where(img < 0.01, 0, img)
            img = np.where(img > 0.95, 0, img)
            img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
            self.data[j] = img
        images = self.data[0:self.seq_len, ...]
        images = np.expand_dims(images, -1)
        images = reshape_patch(images, self.config.data.patch)
        temp = np.zeros_like(images)
        return torch.tensor(images).permute(0, 3, 1, 2), temp
    def __len__(self):
        return self.length

class data_set(data.Dataset):
    def __init__(self,train_path,train_label_path, test_path, test_label_path, is_train):
        super(data_set, self).__init__()
        if is_train:
            self.dataset = np.load(train_path)
            self.label = np.load(train_label_path)
            self.length = self.dataset.shape[0]
        else:
            self.dataset = np.load(test_path)
            self.label = np.load(test_label_path)
            self.length = self.dataset.shape[0]
    def __getitem__(self, idx):
        input = self.dataset[idx]
        output = self.label[idx]
        input = torch.from_numpy(input).contiguous().float()
        output = torch.from_numpy(output).contiguous().float()
        return input,output
    def __len__(self):
        return self.length


def rotate_nobound(image, angle , center=None, scale=1.):
    image = np.swapaxes(image, 2, 0)
    image = np.swapaxes(image, 1, 0)
    # image = np.swapaxes(image, )
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)  # perform the rotation

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    rotated = np.swapaxes(rotated, 1, 0)
    rotated = np.swapaxes(rotated, 2, 0)

    # print(np.array(rotated).shape)
    return rotated

class RandomRotate(object):
    def __init__(self, angles, bound=False):
        self.angles = angles
    def __call__(self, image):
        do_rotate = random.randint(0, 2)
        if do_rotate:
            angle = np.random.uniform(self.angles[0], self.angles[1])
            image = rotate_nobound(image, angle)


        # print(image.shape)
        return image

if __name__ == '__main__':
    with open('../configs/weather_round.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)


    # nn = data_set(train_path='/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_2/video_samples/videos/matrix1_train.npy',
    #                     train_label_path='/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_2/video_samples/videos/matrix2_train.npy',
    #                     test_path='/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_2/video_samples/videos/matrix1_test.npy',
    #                     test_label_path='/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_2/video_samples/videos/matrix2_test.npy',
    #                     is_train=True,)
    # # nn = VaildSet(vaild_path='data/vaild.npy')
    #
    # outs = nn.__getitem__(0)
    # print(outs[1].shape, nn.__len__())



    #
    # nn = MovingRadar_V3(config, seq_len=30,)
    # start = time.time()
    # outs = nn.__getitem__(77)
    # data = outs[0]
    # print(data.shape, data.min(), data.max())
    # data = data.permute(0, 2, 3, 1)
    # print(data.shape)
    # data = np.squeeze(reshape_patch_back(data.numpy(), config.data.patch))
    # print(data.shape)

    # mm1 = VaildSet_v3(config, root='/data04/yuqy_1/train_data/',
    #                  train_path='Z9072_data_max70.npy',
    #                  n_frames_input=10, )
    # out = mm1.__getitem__(13)
    # print(out[0].shape, out[0].max(), out[0].min(),type(out[0]))
    #
    #
    # mm1 = VaildSet(config, root='/data04/yuqy_1/train_data/',
    #                  train_path='QZJSY_traindata_1644samples_max70.npy',
    #                  n_frames_input=20, )
    # out = mm1.__getitem__(13)
    # print(out[0].shape, out[0].max(), out[0].min(),type(out[0]))
#
    # mm1 = Weather(config, root='/data02/dataset/weather_round1_test/input/*.pt')
    # out = mm1.__getitem__(13)
    # print(out[0].shape, out[0].max(), out[0].min(),type(out[0]))

    mm1 = VaildSet_v2(config, root='/data02/dataset/2007',
                         train_path='data_2007_160.npy',)
    out = mm1.__getitem__(13)
    print(out[0].shape, out[0].max(), out[0].min(),type(out[0]))