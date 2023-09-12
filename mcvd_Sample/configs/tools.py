import imageio
import numpy as np
import torch
import os
import cv2
import argparse
import yaml
from PIL import Image
from math import ceil
from torchvision.utils import make_grid, save_image

def save_pred_gif(gif_frames_cond, gif_frames_pred, output_path):

    color_mappings = [[255, 255, 255], [204, 254, 253], [153, 253, 251], [102, 252, 249], [51, 251, 247], [1, 250, 246],
                      [1, 232, 246], [1, 214, 246], [1, 196, 246], [1, 178, 246], [1, 160, 246], [0, 175, 244],
                      [0, 190, 242], [0, 205, 240], [0, 220, 238], [0, 236, 236], [0, 232, 188], [0, 228, 141],
                      [0, 224, 94], [0, 220, 47], [0, 216, 0], [0, 201, 0], [0, 187, 0], [0, 172, 0], [0, 158, 0],
                      [1, 144, 0], [51, 166, 0], [102, 188, 0], [153, 210, 0], [204, 232, 0], [255, 255, 0],
                      [250, 242, 0], [245, 229, 0], [240, 217, 0], [235, 204, 0], [231, 192, 0], [235, 182, 0],
                      [240, 172, 0], [245, 163, 0], [250, 153, 0], [255, 144, 0], [255, 115, 0], [255, 86, 0],
                      [255, 57, 0], [255, 28, 0], [255, 0, 0], [246, 0, 0], [238, 0, 0], [230, 0, 0], [222, 0, 0],
                      [214, 0, 0], [209, 0, 0], [205, 0, 0], [200, 0, 0], [196, 0, 0], [192, 0, 0], [204, 0, 48],
                      [217, 0, 96], [229, 0, 144], [242, 0, 192], [173, 144, 240]]
    level_ranges = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]

    # Define a function to map colors based on levels
    def map_colors(frame, mappings, ranges):
        mapped_frame = np.zeros_like(frame)
        for i in range(len(ranges)):
            if i == 0:
                mask = (frame[:, :, 0] >= 0) & (frame[:, :, 0] <= ranges[i])
            elif i == len(ranges):
                mask = frame[:, :, 0] > ranges[-1]
            else:
                mask = (frame[:, :, 0] > ranges[i - 1]) & (frame[:, :, 0] <= ranges[i])
            mapped_frame[mask] = mappings[i]
        return mapped_frame
    # mapped_cond_frames = [map_colors(frame, color_mappings, level_ranges) for frame in gif_frames_cond]
    mapped_pred_frames = [map_colors(frame, color_mappings, level_ranges) for frame in gif_frames_pred]
    # combined_frames = [*mapped_cond_frames, *mapped_pred_frames]
    combined_frames = [ *mapped_pred_frames]
    imageio.mimwrite(output_path, combined_frames, fps=2)


    # imageio.mimwrite(output_path, [*gif_frames_cond, *gif_frames_pred], fps=4)
#

# Stretch out multiple frames horizontally

def stretch_image(X, ch, imsizeh,imsizew):
    return X.reshape(len(X), -1, ch, imsizeh, imsizew).permute(0, 2, 1, 4, 3).reshape(len(X), -1, imsizew*ch, imsizeh).permute(0, 1, 3, 2)

def save_pred(pred, real, train, cond, config, ckpt, videos_pred_path, video_folder_videos_pred_path, videos_stretch_pred_path, videos_stretch_pred_test_path):
    if train:
        torch.save({"cond": cond, "pred": pred, "real": real}, videos_pred_path,)
    else:
        torch.save({"cond": cond, "pred": pred, "real": real},video_folder_videos_pred_path)
    cond_im = stretch_image(cond, config.data.num_frames_cond, config.data.image_height, config.data.image_width)
    pred_im = stretch_image(pred, config.sampling.num_frames_pred, config.data.image_height, config.data.image_width)
    real_im = stretch_image(real, config.sampling.num_frames_pred, config.data.image_height, config.data.image_width)
    padding_hor = 0.5 * torch.ones(*real_im.shape[:-1], 2)
    real_data = torch.cat([cond_im, padding_hor, real_im], dim=-1)
    pred_data = torch.cat([0.5 * torch.ones_like(cond_im), padding_hor, pred_im], dim=-1)
    padding_ver = 0.5 * torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
    data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
    # Save
    nrow = ceil(np.sqrt(
        (config.data.num_frames_cond + config.sampling.num_frames_pred) * pred.shape[0]) / (
                            config.data.num_frames_cond + config.sampling.num_frames_pred))
    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
    if train:
        save_image(image_grid,videos_stretch_pred_path)
    else:
        save_image(image_grid, videos_stretch_pred_test_path)



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



def reshape_patch(img_tensor, patch_size):
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[np.newaxis, ...]
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]


    a = np.reshape(img_tensor, [batch_size, seq_length,
                                int(img_height / patch_size), patch_size,
                                int(img_width / patch_size), patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  int(img_height / patch_size),
                                  int(img_width / patch_size),
                                  patch_size * patch_size * num_channels])
    patch_tensor = np.squeeze(patch_tensor)
    if patch_tensor.ndim == 3:
        patch_tensor = patch_tensor[..., np.newaxis]
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    if patch_tensor.ndim == 4:
        patch_tensor = patch_tensor[np.newaxis, ...]
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels / (patch_size * patch_size)

    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  int(img_channels)])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                int(img_channels)])
    return img_tensor

def reshape_patch_torch(img_tensor, patch_size):
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(0)
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    num_channels = img_tensor.shape[4]

    a = img_tensor.reshape(batch_size, seq_length,
                           int(img_height / patch_size), patch_size,
                           int(img_width / patch_size), patch_size,
                           num_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    patch_tensor = b.reshape(batch_size, seq_length,
                             int(img_height / patch_size),
                             int(img_width / patch_size),
                             patch_size * patch_size * num_channels)
    return patch_tensor.squeeze()
def reshape_patch_back_torch(patch_tensor, patch_size):
    if patch_tensor.ndim == 4:
        patch_tensor = patch_tensor.unsqueeze(0)

    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    channels = patch_tensor.shape[4]
    img_channels = channels // (patch_size * patch_size)

    a = patch_tensor.reshape(batch_size, seq_length, patch_height, patch_width,
                             patch_size, patch_size, img_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    img_tensor = b.reshape(batch_size, seq_length, patch_height * patch_size,
                           patch_width * patch_size, img_channels)

    return img_tensor




def imgwrite_( data, config,):
    resultpath = '/data03/wuqiliang/code/mcvd_Sample/data/radar'
    namelist = np.load(os.path.join(resultpath, 'index_b.npy'))
    data = data.reshape(data.shape[0], -1, config.data.channels, config.data.image_height, config.data.image_width).permute(0, 1, 3, 4, 2)
    real_gif = []
    data = np.squeeze(reshape_patch_back(data.numpy(), config.data.patch))
    # data = data[:, :-2, ...]
    for dir in range(len(namelist)):
        dirpath = os.path.join(resultpath, 'project/')
        os.makedirs(dirpath, exist_ok=True)
        for i in range(data.shape[1]):
            png = data[dir][i] * 250
            real_gif.append(png.astype('uint8'))
            png = cv2.resize(png, (815, 430), interpolation=cv2.INTER_AREA)
            png = np.where(png < 2.7, 0, png)
            # if i > 33:
            #     png = np.where(png > 0, 0, 0)
            print(os.path.join(dirpath, 'pre_' + str(int(namelist[dir])) + '_' + str(i + 13) + '.png'))
            cv2.imwrite(os.path.join(dirpath, 'pre_'+str(int(namelist[dir]))+'_'+str(i+13)+'.png'), png)
        for _ in range(36 - data.shape[1]):
            i += 1
            png = np.zeros_like(png)
            print(os.path.join(dirpath, 'pre_' + str(int(namelist[dir])) + '_' + str(i + 13) + '.png'))
            cv2.imwrite(os.path.join(dirpath, 'pre_'+str(int(namelist[dir]))+'_'+str(i+13)+'.png'), png)

        # imageio.mimwrite(os.path.join(dirpath, "result.gif"), real_gif, fps=4)
        # print(data.shape)


if __name__ == '__main__':
    with open('../configs/radar256.yml', 'r') as f:
        # with open('../configs/test.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    path = '/data03/wuqiliang/code/mcvd_Sample/videos/resultssss.pt'
    data = torch.load(path)
    print(data.shape)
    data = data.reshape(data.shape[0], -1, config.data.channels, config.data.image_height,
                          config.data.image_width).permute(0, 1, 3, 4, 2)
    data = np.squeeze(reshape_patch_back(data.numpy(), config.data.patch))
    np.save('/data03/wuqiliang/code/mcvd_Sample/videos/pred20.npy',data)
    # print(data.shape)
    imgwrite_(data, config)



