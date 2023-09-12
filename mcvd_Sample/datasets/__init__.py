import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.stochastic_moving_mnist import MovingRadar_V3,MovingRadar_V4,MovingRadar_V5,VaildSet, VaildSet_v2, Weather, VaildSet_v3

from torch.utils.data import Subset
DATASETS = ['MOVINGRADAR_V3','MOVINGRADAR_V4','MOVINGRADAR_V5', 'WEATHER', 'VaildSet_v3']

def get_dataloaders(data_path, config):
    dataset, test_dataset = get_dataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=True,
                             num_workers=config.data.num_workers, drop_last=True)
    return dataloader, test_loader


def get_dataset(config, video_frames_pred=0, valid=False):
    assert config.data.dataset.upper() in DATASETS, \
        f"datasets/__init__.py: dataset can only be in {DATASETS}! Given {config.data.dataset.upper()}"
    dataset, test_dataset = None, None
    if not valid:
        if config.data.dataset.upper() == "MOVINGRADAR_V3":
            seq_len = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
            # dataset = MovingRadar_V3(config, seq_len=seq_len, is_train=True,)
            # test_dataset = MovingRadar_V3(config, seq_len=seq_len, is_train=False)
            # test_dataset = VaildSet(config, root='/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/user_data/data/',
            #          train_path='valid_radar_256.npy',
            #          n_frames_input=seq_len,)
            # test_dataset = VaildSet(config, root='/data04/yuqy_1/train_data/',
            #          train_path='sanya.npy',
            #          n_frames_input=seq_len, )

            dataset = VaildSet_v3(config, root='/data04/yuqy_1/train_data/',
                     train_path='Z9072_data_max70.npy',
                     n_frames_input=seq_len, )
            # test_dataset = VaildSet_v3(config, root='/data04/yuqy_1/train_data/',
            #          train_path='Z9072_data_max70.npy',
            #          n_frames_input=seq_len, )
            test_dataset =  VaildSet(config, root='/data04/yuqy_1/predRNN_test_data/',
                   train_path='Z9898_test_data_0716.npy',
                   n_frames_input=seq_len,)

        if config.data.dataset.upper() == "WEATHER":
            seq_len = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
            dataset = VaildSet_v2(config, root='/data02/dataset/weather',
                     train_path='data_5.npy',
                     n_frames_input=seq_len, )
            test_dataset = VaildSet_v2(config, root='/data02/dataset/weather',
                     train_path='data_2007_160_5.npy',
                   n_frames_input=seq_len, lens=1400)

        if config.data.dataset.upper() == "MOVINGRADAR_V5":
            seq_len = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
            dataset = MovingRadar_V5(config, seq_len=seq_len, is_train=True)
            test_dataset = MovingRadar_V5(config, seq_len=seq_len, is_train=False)
    else:
        # if config.data.dataset_valid == 'tianchi':
            # seq_len = config.data.num_frames_cond
            # dataset = VaildSet(config, root='/data03/wuqiliang/Dataset/vae/radar_nowcasting-main/user_data/data/',
            #          train_path='valid_radar_256.npy',
            #          n_frames_input=seq_len,)
            # test_dataset = None
        if config.data.dataset_valid == 'tianchi_weather':
            seq_len = config.data.num_frames_cond
            dataset = Weather(config, root='/data02/dataset/weather_round1_test/input/*.pt')
            test_dataset = None
        else:
            seq_len = config.data.num_frames_cond
            dataset = MovingRadar_V4(config, seq_len=seq_len)
            test_dataset = None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def normalize(data, mean= 18.709646, std= 33.578693):
    normalized_data = (data - mean) / std
    return normalized_data

def de_normalize(normalized_data, mean= 18.709646, std= 33.578693):
    denormalized_data = normalized_data * std + mean
    return denormalized_data


def min_max_scale(data, min=-20, max=20):
    return (data - min) / (max- min)

def inverse_min_max_scale(inverse_data, min=-20, max=20):
    inv_data = inverse_data * (max - min) + min
    return inv_data

def data_transform(config, X):
    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        # X = logit_transform(X)
        X = X
    elif config.data.zero_mean_normalization:
        X = normalize(X)

    elif config.data.zero_mean_normalization_255:
        X = normalize(X, mean= 0.07337143, std= 0.13166589)
        # X = 2 * X - 1.
    elif config.data.max_min_mean_normalization:
        X = min_max_scale(X)
    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]
    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]
    if config.data.logit_transform:
        # X = torch.sigmoid(X)
        X = X
        return X
    elif config.data.rescaled:
        X = (X + 1.) / 2.
        return torch.clamp(X, 0.0, 1.0)
    elif config.data.zero_mean_normalization:
        X = de_normalize(X)
        return torch.clamp(X, 0.0, 1.0)
    elif config.data.zero_mean_normalization_255:
        X = de_normalize(X, mean= 0.07337143, std= 0.13166589)
        # X = (X + 1.) / 2.
        return X
    elif config.data.max_min_mean_normalization:
        X = inverse_min_max_scale(X)
        return X



