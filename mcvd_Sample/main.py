import time
from configs.configration import parse_args_and_config
from models.ema import EMAHelper
import matplotlib
matplotlib.use('Agg')
from configs.tools import reshape_patch_back,reshape_patch, reshape_patch_back_torch, reshape_patch_torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sys
import scipy.stats as st
from math import ceil
from functools import partial
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from datasets import get_dataset, data_transform, inverse_data_transform
from models import ddpm_sampler
import logging
import torchvision.transforms as Transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from cv2 import putText
import imageio
import datetime
import psutil
import yaml
from losses import get_optimizer, warmup_lr
import math
from losses.dsm import anneal_dsm_score_estimation
from multiprocessing import Process
from  configs.tools import  save_pred_gif, save_pred
from models.model_attnunet import AttU_Net
def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024 ** 3

def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0

def conditioning_fn(config, X, num_frames_pred=0, conditional=True, vaild=False):
    imsizeh = config.data.image_height
    imsizew = config.data.image_width
    cond = config.data.num_frames_cond
    if not vaild:
        pred = num_frames_pred
        pred_frames = X[:, cond:cond+pred].reshape(len(X), -1, imsizeh, imsizew)
    else:
        pred_frames = None
    cond_frames = X[:, :cond].reshape(len(X), -1, imsizeh, imsizew)
    cond_mask = None
    return pred_frames, cond_frames, cond_mask



def get_model(config):

    arch = getattr(config.model, 'arch', 'ncsn')

    if arch == 'unetmore':
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        return UNetMore_DDPM(config).to(config.device)#.to(memory_format=torch.channels_last).to(config.device)
    else:
        Exception("arch is not valid [unetmore]")

# def pre_stretch_image(X, ch, imsize, num_frames=4 ):
#
#     if len(X.shape) == 5:
#         X = torch.squeeze(X)
#
#     return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 1, 2, 4, 3).reshape(len(X), -1, ch * imsize,
#                                                                                     imsize).permute(0, 1, 3, 2)


# def stretch_image(X, ch, imsize):
#     return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)

def stretch_image(X, ch, imsizeh,imsizew):
    return X.reshape(len(X), -1, ch, imsizeh, imsizew).permute(0, 2, 1, 4, 3).reshape(len(X), -1, imsizew*ch, imsizeh).permute(0, 1, 3, 2)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)


class NCSNRunner():
    def __init__(self, args, config, config_uncond):
        self.args = args
        self.config = config
        self.config_uncond = config_uncond
        self.version = getattr(self.config.model, 'version', "SMLD")
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        self.get_mode()

        self.calc_ssim = getattr(self.config.sampling, "ssim", False)
        self.calc_fvd = getattr(self.config.sampling, "fvd", False)
        self.calc_fvd1, self.calc_fvd2,  self.calc_fvd3 = False, False, False
        if self.calc_fvd:
            if self.condp == 0.0 and self.futrf == 0:  # (1) Prediction
                self.calc_fvd1 = self.condf + self.config.sampling.num_frames_pred >= 10




    def get_mode(self):
        self.condf, self.condp = self.config.data.num_frames_cond, getattr(self.config.data, "prob_mask_cond", 0.0)
        self.futrf, self.futrp = getattr(self.config.data, "num_frames_future", 0), getattr(self.config.data,
                                                                                            "prob_mask_future", 0.0)
        self.prob_mask_sync = getattr(self.config.data, "prob_mask_sync", False)
        if not getattr(self.config.sampling, "ssim", False):
            if getattr(self.config.sampling, "fvd", False):
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, "three"
            else:
                self.mode_pred, self.mode_interp, self.mode_gen = None, None, None
        elif self.condp == 0.0 and self.futrf == 0:  # (1) Prediction
            self.mode_pred, self.mode_interp, self.mode_gen = "one", None, None

    def get_sampler(self):
        # Sampler
        sampler = None
        if self.version == "DDPM":
            sampler = partial(ddpm_sampler, config=self.config)

        return sampler
    def write_to_yaml(self, yaml_file, my_dict):
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                old_dict = yaml.load(f, Loader=yaml.FullLoader)
            for key in my_dict.keys():
                old_dict[key] = my_dict[key]
            my_dict = {}
            for key in sorted(old_dict.keys()):
                my_dict[key] = old_dict[key]
        with open(yaml_file, 'w') as f:
            yaml.dump(my_dict, f, default_flow_style=False)
    def load_meters(self):
        meters_pkl = os.path.join(self.args.log_path, 'meters.pkl')
        if not os.path.exists(meters_pkl):
            print(f"{meters_pkl} does not exist! Returning.")
            return False
        with open(meters_pkl, "rb") as f:
            a = pickle.load(f)
        # Load
        self.epochs = a['epochs']
        self.losses_train = a['losses_train']
        self.losses_test = a['losses_test']
        self.lr_meter = a['lr_meter']
        self.grad_norm = a['grad_norm']
        self.time_train = a['time_train']
        self.time_train_prev = a['time_train'].val or 0
        self.time_elapsed = a['time_elapsed']
        self.time_elapsed_prev = a['time_elapsed'].val or 0
        try:
            self.mses = a['mses']
            self.psnrs = a['psnrs']
            self.ssims = a['ssims']
            self.lpipss = a['lpips']
            self.fvds = a['fvds']
            self.best_mse = a['best_mse']
            self.best_psnr = a['best_psnr']
            self.best_ssim = a['best_ssim']
            self.best_lpips = a['best_lpips']
            self.best_fvd = a['best_fvd']
        except:
            self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.best_mse = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
        try:
            self.mses2 = a['mses2']
            self.psnrs2 = a['psnrs2']
            self.ssims2 = a['ssims2']
            self.lpipss2 = a['lpips2']
            self.fvds2 = a['fvds2']
            self.fvds3 = a['fvds3']
            self.best_mse2 = a['best_mse2']
            self.best_psnr2 = a['best_psnr2']
            self.best_ssim2 = a['best_ssim2']
            self.best_lpips2 = a['best_lpips2']
            self.best_fvd2 = a['best_fvd2']
            self.best_fvd3 = a['best_fvd3']
        except:
            self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2 = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.fvds3 = RunningAverageMeter()
            self.best_mse2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd3 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                              'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
        return True


    def init_meters(self):
        success = self.load_meters()
        # success = False
        if not success:
            self.epochs = RunningAverageMeter()
            self.losses_train, self.losses_test = RunningAverageMeter(), RunningAverageMeter()
            self.lr_meter, self.grad_norm = RunningAverageMeter(), RunningAverageMeter()
            self.time_train, self.time_elapsed = RunningAverageMeter(), RunningAverageMeter()
            self.time_train_prev = self.time_elapsed_prev = 0
            self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.mses2, self.psnrs2, self.ssims2, self.lpipss2, self.fvds2 = RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter(), RunningAverageMeter()
            self.fvds3 = RunningAverageMeter()
            self.best_mse = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_mse2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_psnr2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_ssim2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_lpips2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd2 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}
            self.best_fvd3 = {'ckpt': -1, 'mse': math.inf, 'psnr': -math.inf, 'ssim': -math.inf, 'lpips': math.inf, 'fvd': math.inf,
                             'mse2': math.inf, 'psnr2': -math.inf, 'ssim2': -math.inf, 'lpips2': math.inf, 'fvd2': math.inf, 'fvd3': math.inf}

    def savefig(self, path, bbox_inches='tight', pad_inches=0.1):
        try:
            plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(sys.exc_info()[0])
    def plot_graphs(self):
        # Losses
        plt.plot(self.losses_train.steps, self.losses_train.vals, label='Train')
        plt.plot(self.losses_test.steps, self.losses_test.vals, label='Test')
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, 'loss.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'loss_log.png'))
        plt.clf()
        plt.close()
        # Epochs
        plt.plot(self.losses_train.steps, self.epochs.vals)
        plt.xlabel("Steps")
        plt.ylabel("Epochs")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'epochs.png'))
        plt.clf()
        plt.close()
        # LR
        plt.plot(self.losses_train.steps, self.lr_meter.vals)
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'lr.png'))
        plt.clf()
        plt.close()
        # Grad Norm
        plt.plot(self.losses_train.steps, self.grad_norm.vals)
        plt.xlabel("Steps")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'grad.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'grad_log.png'))
        plt.clf()
        plt.close()
        # Time train
        plt.plot(self.losses_train.steps, self.time_train.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_train.png'))
        plt.clf()
        plt.close()
        # Time elapsed
        plt.plot(self.losses_train.steps[:len(self.time_elapsed.vals)], self.time_elapsed.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_elapsed.png'))
        plt.clf()
        plt.close()
    def convert_time_stamp_to_hrs(self, time_day_hr):
        time_day_hr = time_day_hr.split(",")
        if len(time_day_hr) > 1:
            days = time_day_hr[0].split(" ")[0]
            time_hr = time_day_hr[1]
        else:
            days = 0
            time_hr = time_day_hr[0]
        # Hr
        hrs = time_hr.split(":")
        return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600

    def plot_video_graphs_single(self, name, mses, psnrs, ssims, lpipss, fvds, calc_fvd,
                                 best_mse, best_psnr, best_ssim, best_lpips, best_fvd):
        # MSE
        plt.plot(mses.steps, mses.vals)
        if best_mse['ckpt'] > -1:
            plt.scatter(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], color='k')
            plt.text(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], f"{mses.vals[mses.steps.index(best_mse['ckpt'])]:.04f}\n{best_mse['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"mse_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"mse_{name}_log.png"))
        plt.clf()
        plt.close()
        # PSNR
        plt.plot(mses.steps, psnrs.vals)
        if best_psnr['ckpt'] > -1:
            plt.scatter(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], color='k')
            plt.text(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], f"{psnrs.vals[mses.steps.index(best_psnr['ckpt'])]:.04f}\n{best_psnr['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("PSNR")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"psnr_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"psnr_{name}_log.png"))
        plt.clf()
        plt.close()
        # SSIM
        plt.plot(mses.steps, ssims.vals)
        if best_ssim['ckpt'] > -1:
            plt.scatter(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], color='k')
            plt.text(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], f"{ssims.vals[mses.steps.index(best_ssim['ckpt'])]:.04f}\n{best_ssim['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("SSIM")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"ssim_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"ssim_{name}_log.png"))
        plt.clf()
        plt.close()
        # LPIPS
        plt.plot(mses.steps, lpipss.vals)
        if best_lpips['ckpt'] > -1:
            plt.scatter(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], color='k')
            plt.text(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], f"{lpipss.vals[mses.steps.index(best_lpips['ckpt'])]:.04f}\n{best_lpips['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("LPIPS")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, f"lpips_{name}.png"))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, f"lpips_{name}_log.png"))
        plt.clf()
        plt.close()
        # FVD
        if calc_fvd:
            plt.plot(mses.steps, fvds.vals)
            if best_fvd['ckpt'] > -1:
                plt.scatter(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], color='k')
                plt.text(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], f"{fvds.vals[mses.steps.index(best_fvd['ckpt'])]:.04f}\n{best_fvd['ckpt']}", c='r')
            plt.xlabel("Steps")
            plt.ylabel("FVD")
            plt.grid(True)
            plt.grid(visible=True, which='minor', axis='y', linestyle='--')
            # plt.legend(loc='upper right')
            self.savefig(os.path.join(self.args.log_path, f"fvd_{name}.png"))
            plt.yscale("log")
            self.savefig(os.path.join(self.args.log_path, f"fvd_{name}_log.png"))
            plt.clf()
            plt.close()


    def plot_video_graphs(self):
        # Pred
        if self.mode_pred is not None:
            self.plot_video_graphs_single("pred",
                                          self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred, self.calc_fvd_pred,
                                          self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred)
    def save_meters(self):
        meters_pkl = os.path.join(self.args.log_path, 'meters.pkl')
        with open(meters_pkl, "wb") as f:
            pickle.dump({
                'epochs': self.epochs,
                'losses_train': self.losses_train,
                'losses_test': self.losses_test,
                'lr_meter' : self.lr_meter,
                'grad_norm' : self.grad_norm,
                'time_train': self.time_train,
                'time_elapsed': self.time_elapsed,
                'mses': self.mses,
                'psnrs': self.psnrs,
                'ssims': self.ssims,
                'lpips': self.lpipss,
                'fvds': self.fvds,
                'best_mse': self.best_mse,
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_lpips': self.best_lpips,
                'best_fvd': self.best_fvd,
                'mses2': self.mses2,
                'psnrs2': self.psnrs2,
                'ssims2': self.ssims2,
                'lpips2': self.lpipss2,
                'fvds2': self.fvds2,
                'best_mse2': self.best_mse2,
                'best_psnr2': self.best_psnr2,
                'best_ssim2': self.best_ssim2,
                'best_lpips2': self.best_lpips2,
                'best_fvd2': self.best_fvd2,
                'best_fvd3': self.best_fvd3,
                },
                f, protocol=pickle.HIGHEST_PROTOCOL)



    def train(self):
        dataset, test_dataset = get_dataset(self.config, video_frames_pred=self.config.data.num_frames,)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_height * self.config.data.image_width * self.config.data.channels

        # tb_logger = self.config.tb_logger
        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        optimizer = get_optimizer(self.config, scorenet.parameters())

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            logging.info(f"Number of GPUs : {num_devices}")
            for i in range(num_devices):
                logging.info(torch.cuda.get_device_properties(i))
        else:
            logging.info(f"Running on CPU!")

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)

        def init_weights(tensor):
            with torch.no_grad():
                tensor.uniform_(-1, 1)
            return tensor
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'))
            # states = torch.load("/data03/wuqiliang/code/mcvd_Sample/radar_256_num_14/logs/samples/checkpoint_0.pt")
            # state[0]['module.unet.all_modules.2.weight'] = init_weights(torch.empty(128, 96, 3, 3))
            # state[0]['module.unet.all_modules.39.weight'] = init_weights(torch.empty(48, 128, 3, 3))
            # state[0]['module.unet.all_modules.39.bias'] = init_weights(torch.empty(48))
            scorenet.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resuming training from checkpoint.pt in {self.args.log_path} at epoch {start_epoch}, step {step}.")

            net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        cond, test_cond = None, None
        # Initialize meters
        self.init_meters()

        # Initial samples
        n_init_samples = min(36, self.config.training.batch_size)
        init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
        if self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            if getattr(self.config.model, 'gamma', False):
                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                init_samples = z - used_k*used_theta # we don't scale here
            else:
                init_samples = torch.randn(init_samples_shape, device=self.config.device)


        self.total_train_time = 0
        self.start_time = time.time()

        early_end = False
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for batch, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                lr = warmup_lr(optimizer, step, getattr(self.config.optim, 'warmup', 0), self.config.optim.lr)
                scorenet.train()
                step += 1
                # Data
                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                X, cond, cond_mask = conditioning_fn(self.config, X, num_frames_pred=self.config.data.num_frames,
                                                     conditional=conditional)
                itr_start = time.time()
                loss = anneal_dsm_score_estimation(scorenet, X, labels=None, cond=cond, cond_mask=cond_mask,
                                                   loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                   gamma=getattr(self.config.model, 'gamma', False),
                                                   L1=getattr(self.config.training, 'L1', False),
                                                   all_frames=getattr(self.config.model, 'output_all_frames', False))
                # tb_logger.add_scalar('loss', loss, global_step=step)
                # tb_hook()

                # Optimize
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(scorenet.parameters(), getattr(self.config.optim, 'grad_clip', np.inf))
                optimizer.step()

                # Training time
                itr_time = time.time() - itr_start
                self.total_train_time += itr_time
                self.time_train.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.total_train_time))) + self.time_train_prev)

                # Record
                self.losses_train.update(loss.item(), step)
                self.epochs.update(epoch + (batch + 1)/len(dataloader))
                self.lr_meter.update(lr)
                self.grad_norm.update(grad_norm.item())
                if step == 1 or step % getattr(self.config.training, "log_freq", 1) == 0:
                    logging.info("elapsed: {}, train time: {:.04f}, mem: {:.03f}GB, GPUmem: {:.03f}GB, step: {}, lr: {:.06f}, grad: {:.04f}, loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        self.time_train.val, get_proc_mem(), get_GPU_mem(), step, lr, grad_norm, loss.item()))

                if self.config.model.ema:
                    ema_helper.update(scorenet)

                if step >= self.config.training.n_iters:
                    early_end = True
                    break

                # Save model
                if (step % 1000 == 0 and step != 0) or step % self.config.training.snapshot_freq == 0:
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    print(os.path.join(self.args.log_path, 'checkpoint.pt'))
                    logging.info(f"Saving checkpoint.pt in {self.args.log_path}")
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))
                    if step % self.config.training.snapshot_freq == 0:
                        ckpt_path = os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step))
                        logging.info(f"Saving {ckpt_path}")
                        torch.save(states, ckpt_path)

                test_scorenet = None
                # Get test_scorenet
                if step == 1 or step % self.config.training.val_freq == 0 or (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    if self.config.model.ema:
                        test_scorenet = ema_helper.ema_copy(scorenet)
                    else:
                        test_scorenet = scorenet

                    test_scorenet.eval()

                # Validation
                if step == 1 or step % self.config.training.val_freq == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    test_X, test_cond, test_cond_mask = conditioning_fn(self.config, test_X, num_frames_pred=self.config.data.num_frames,
                                                                        conditional=conditional)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_scorenet, test_X, labels=None, cond=test_cond, cond_mask=test_cond_mask,
                                                                    loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                                    gamma=getattr(self.config.model, 'gamma', False),
                                                                    L1=getattr(self.config.training, 'L1', False),
                                                                    all_frames=getattr(self.config.model, 'output_all_frames', False))
                    # tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                    # test_tb_hook()
                    self.losses_test.update(test_dsm_loss.item(), step)
                    logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, test_loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        step, get_proc_mem(), get_GPU_mem(), test_dsm_loss.item()))

                    # Plot graphs
                    try:
                        plot_graphs_process.join()
                    except:
                        pass
                    plot_graphs_process = Process(target=self.plot_graphs)
                    plot_graphs_process.start()

                # Sample from model
                if (step % self.config.training.snapshot_freq == 0 or step % self.config.training.sample_freq == 0) and self.config.training.snapshot_sampling:

                    logging.info(f"Saving images in {self.args.log_sample_path}")

                    # Calc video metrics with max_data_iter=1
                    if conditional and step % self.config.training.snapshot_freq == 0 and self.config.training.snapshot_sampling: # only at snapshot_freq, not at sample_freq

                        vid_metrics = self.video_gen(scorenet=test_scorenet, ckpt=step, train=True)

                        if 'mse' in vid_metrics.keys():
                            self.mses.update(vid_metrics['mse'], step)
                            self.psnrs.update(vid_metrics['psnr'])
                            self.ssims.update(vid_metrics['ssim'])
                            self.lpipss.update(vid_metrics['lpips'])
                            if vid_metrics['mse'] < self.best_mse['mse']:
                                self.best_mse = vid_metrics
                            if vid_metrics['psnr'] > self.best_psnr['psnr']:
                                self.best_psnr = vid_metrics
                            if vid_metrics['ssim'] > self.best_ssim['ssim']:
                                self.best_ssim = vid_metrics
                            if vid_metrics['lpips'] < self.best_lpips['lpips']:
                                self.best_lpips = vid_metrics
                            if self.calc_fvd1:
                                self.fvds.update(vid_metrics['fvd'])
                                if vid_metrics['fvd'] < self.best_fvd['fvd']:
                                    self.best_fvd = vid_metrics

                        if 'mse2' in vid_metrics.keys():
                            self.mses2.update(vid_metrics['mse2'], step)
                            self.psnrs2.update(vid_metrics['psnr2'])
                            self.ssims2.update(vid_metrics['ssim2'])
                            self.lpipss2.update(vid_metrics['lpips2'])
                            if vid_metrics['mse2'] < self.best_mse2['mse2']:
                                self.best_mse2 = vid_metrics
                            if vid_metrics['psnr2'] > self.best_psnr2['psnr2']:
                                self.best_psnr2 = vid_metrics
                            if vid_metrics['ssim2'] > self.best_ssim2['ssim2']:
                                self.best_ssim2 = vid_metrics
                            if vid_metrics['lpips2'] < self.best_lpips2['lpips2']:
                                self.best_lpips2 = vid_metrics
                            if self.calc_fvd2:
                                self.fvds2.update(vid_metrics['fvd2'])
                                if vid_metrics['fvd2'] < self.best_fvd2['fvd2']:
                                    self.best_fvd2 = vid_metrics

                        if self.calc_fvd3:
                            self.fvds3.update(vid_metrics['fvd3'], step)
                            if vid_metrics['fvd3'] < self.best_fvd3['fvd3']:
                                self.best_fvd3 = vid_metrics

                        # Show best results for every metric

                        if self.condp == 0.0 and self.futrf == 0:                           # (1) Prediction
                            self.mses_pred, self.psnrs_pred, self.ssims_pred, self.lpipss_pred, self.fvds_pred = self.mses, self.psnrs, self.ssims, self.lpipss, self.fvds
                            self.best_mse_pred, self.best_psnr_pred, self.best_ssim_pred, self.best_lpips_pred, self.best_fvd_pred, self.calc_fvd_pred = self.best_mse, self.best_psnr, self.best_ssim, self.best_lpips, self.best_fvd, self.calc_fvd1

                        format_p = lambda dd : ", ".join([f"{k}:{v:.4f}" if k != 'ckpt' and k != 'preds_per_test' else f"{k}:{v:7d}" if k == 'ckpt' else f"{k}:{v:3d}" for k, v in dd.items()])
                        if self.mode_pred is not None:
                            logging.info(f"PRED: {self.mode_pred}")
                            logging.info(f"Best-MSE   pred - {format_p(self.best_mse_pred)}")
                            logging.info(f"Best-PSNR  pred - {format_p(self.best_psnr_pred)}")
                            logging.info(f"Best-SSIM  pred - {format_p(self.best_ssim_pred)}")
                            logging.info(f"Best-LPIPS pred - {format_p(self.best_lpips_pred)}")
                            if self.calc_fvd_pred:
                                logging.info(f"Best-FVD   pred - {format_p(self.best_fvd_pred)}")
                        # Plot video graphs
                        try:
                            plot_video_graphs_process.join()
                        except:
                            pass
                        plot_video_graphs_process = Process(target=self.plot_video_graphs)
                        plot_video_graphs_process.start()

                del test_scorenet
                self.time_elapsed.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=(time.time() - self.start_time)))) + self.time_elapsed_prev)
                # Save meters
                if step == 1 or step % self.config.training.val_freq == 0 or step % 1000 == 0 or step % self.config.training.snapshot_freq == 0:
                    self.save_meters()

            if early_end:
                break


        # Save model at the very end
        states = [
            scorenet.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        logging.info(f"Saving checkpoints in {self.args.log_path}")
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step)))
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pt'))
        # Show best results for every metric
        logging.info("Best-MSE - {}".format(", ".join([f"{k}:{self.best_mse[k]}" for k in self.best_mse])))
        logging.info("Best-PSNR - {}".format(", ".join([f"{k}:{self.best_psnr[k]}" for k in self.best_psnr])))
        logging.info("Best-SSIM - {}".format(", ".join([f"{k}:{self.best_ssim[k]}" for k in self.best_ssim])))
        logging.info("Best-LPIPS - {}".format(", ".join([f"{k}:{self.best_lpips[k]}" for k in self.best_lpips])))
        if getattr(self.config.sampling, "fvd", False):
            logging.info("Best-FVD - {}".format(", ".join([f"{k}:{self.best_fvd[k]}" for k in self.best_fvd])))





    def sample(self):
        if self.config.sampling.ckpt_id is None:
            ckpt = "latest"
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
        else:
            ckpt = self.config.sampling.ckpt_id
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pt'),
                                map_location=self.config.device)

        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        scorenet.load_state_dict(states[0], strict=False)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)

        dataset_train, dataset_test = get_dataset(self.config, video_frames_pred=self.config.data.num_frames)
        dataset = dataset_train if getattr(self.config.sampling, "train", False) else dataset_test
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)

        scorenet.eval()

        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        cond = None

        # Future
        future = getattr(self.config.data, "num_frames_future", 0)

        if not self.config.sampling.fid:
            if self.config.sampling.data_init or conditional:
                data_iter = iter(dataloader)
                real, _ = next(data_iter)
                real = real.to(self.config.device)
                real = data_transform(self.config, real)
                real, cond, cond_mask = conditioning_fn(self.config, real, num_frames_pred=self.config.data.num_frames,
                                                        conditional=conditional)
            init_samples_shape = (self.config.sampling.batch_size, self.config.data.channels * self.config.data.num_frames,
                                  self.config.data.image_height, self.config.data.image_width)

            if version == "DDPM" or version == "DDIM" or version == "FPNDM":
                if getattr(self.config.model, 'gamma', False):
                    used_k, used_theta = net.k_cum[0], net.theta_t[0]
                    z = Gamma(torch.full(init_samples_shape, used_k),
                              torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                    z = z - used_k * used_theta
                else:
                    z = torch.randn(init_samples_shape, device=self.config.device)

            # init_samples
            init_samples = z

            # Sampler
            sampler = self.get_sampler()

            all_samples = sampler(init_samples, scorenet, cond=cond, cond_mask=cond_mask,
                                  n_steps_each=self.config.sampling.n_steps_each,
                                  step_lr=self.config.sampling.step_lr, verbose=True,
                                  final_only=self.config.sampling.final_only,
                                  denoise=self.config.sampling.denoise,
                                  subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                  clip_before=getattr(self.config.sampling, 'clip_before', True),
                                  log=True, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')
            if not self.config.sampling.final_only:
                for i, sample in tqdm(enumerate(all_samples), total=len(all_samples),
                                      desc="saving image samples"):
                    sample = sample.reshape(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_height, self.config.data.image_width)

                    sample = inverse_data_transform(self.config, sample)
                    sample = stretch_image(sample, self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
                    nrow = ceil(np.sqrt(
                        self.config.data.num_frames * self.config.sampling.batch_size) / self.config.data.num_frames)
                    image_grid = make_grid(sample, nrow, pad_value=0.5)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{:04d}.png'.format(i)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'sa_{:04d}.pt'.format(i)))

            else:
                sample = all_samples[-1].reshape(all_samples[-1].shape[0],
                                                 self.config.data.channels * self.config.data.num_frames,
                                                 self.config.data.image_height, self.config.data.image_width)
                sample = inverse_data_transform(self.config, sample)

                sample = stretch_image(sample, self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
                nrow = ceil(
                    np.sqrt(self.config.data.num_frames * self.config.sampling.batch_size) / self.config.data.num_frames)
                image_grid = make_grid(sample, nrow, pad_value=0.5)
                save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(ckpt)))
                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pt'.format(ckpt)))

            if conditional:
                real, cond = real.to('cpu'), cond.to('cpu')
                torch.save(real, os.path.join(self.args.image_folder, 'real{}.pt'.format(ckpt)))

                real = stretch_image(inverse_data_transform(self.config, real), self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
                if future > 0:
                    cond, futr = torch.tensor_split(cond, (self.config.data.num_frames_cond * self.config.data.channels,),
                                                    dim=1)
                    futr = stretch_image(inverse_data_transform(self.config, futr), self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
                cond = stretch_image(inverse_data_transform(self.config, cond), self.config.data.num_frames, self.config.data.image_height, self.config.data.image_width)
                padding = 2.5 * torch.ones(len(real), 1, self.config.data.image_height, 2)
                nrow = ceil(np.sqrt((self.config.data.num_frames_cond + self.config.data.num_frames * 2 + future) * self.config.sampling.batch_size) / (
                                        self.config.data.num_frames_cond + self.config.data.num_frames * 2 + future))
                image_grid = make_grid(torch.cat(
                    [cond, padding, real, padding, sample] if future == 0 else [cond, padding, real, padding, sample, futr],
                    dim=-1), nrow=nrow, padding=6, pad_value=0.5)
                save_image(image_grid, os.path.join(self.args.image_folder, 'image_full_grid_{}.png'.format(ckpt)))
                torch.save([real, sample], os.path.join(self.args.image_folder, 'samples_real{}.pt'.format(ckpt)))



    @torch.no_grad()
    def video_gen(self, scorenet=None, ckpt=None, train=False):
        from models.model_attnunet import AttU_Net
        model2 = AttU_Net(10, 10).cuda()
        root_model_path =f'/data03/wuqiliang/code/mcvd_Sample/weather_20/tianchi/weight'
        model_path = os.path.join(root_model_path, 'generator_r48.pth')
        model2.load_state_dict(torch.load(model_path, map_location=self.config.device))
        model2.eval()
        #
        # model1 = AttU_Net(10, 10).cuda()
        # root_model_path =f'/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_1/sanya/weight/'
        # model_path = os.path.join(root_model_path, 'generator_r21.pth')
        # model1.load_state_dict(torch.load(model_path, map_location=self.config.device))
        # model1.eval()
        # Sample n predictions per test data, choose the best among them for each metric
        # calc_ssim = getattr(self.config.sampling, "ssim", False)
        # calc_fvd = getattr(self.config.sampling, "fvd", False)
        # calc_fvd1, calc_fvd2,  calc_fvd3 = False, False, False
        # if calc_fvd:
        #     if self.condp == 0.0 and self.futrf == 0:  # (1) Prediction
        #         calc_fvd1 = self.condf + self.config.sampling.num_frames_pred >= 10
        #         calc_fvd2 = calc_fvd3 = False
        #     if calc_fvd1 or calc_fvd2 or calc_fvd3:
        #         i3d = load_i3d_pretrained(self.config.device)
        # else:
        #     self.calc_fvd1, self.calc_fvd2, self.calc_fvd3 = calc_fvd1, calc_fvd2, calc_fvd3 = False, False, False
        #     if calc_ssim is False:
        #         return {}
        if train:
            assert(scorenet is not None and ckpt is not None)
            max_data_iter = 1   # self.config.sampling.max_data_iter
            preds_per_test = 1  # self.config.sampling.preds_per_test
        else:
            self.start_time = time.time()
            max_data_iter = self.config.sampling.max_data_iter
            preds_per_test = getattr(self.config.sampling, 'preds_per_test', 1)

        if scorenet is None:
            if self.config.sampling.ckpt_id is None:
                ckpt = "latest"
                logging.info(f"Loading ckpt {os.path.join(self.args.log_path, 'checkpoint.pt')}")
                states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
            else:
                ckpt = self.config.sampling.ckpt_id
                ckpt_file = os.path.join(self.args.log_path, f'checkpoint_960000.pt')
                logging.info(f"Loading ckpt {ckpt_file}")
                states = torch.load(ckpt_file, map_location=self.config.device)

            scorenet = get_model(self.config)
            scorenet = torch.nn.DataParallel(scorenet)

            scorenet.load_state_dict(states[0], strict=False)
            scorenet.eval()

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(scorenet)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(scorenet)

        # Collate fn for n repeats
        def my_collate(batch):
            data, _ = zip(*batch)
            data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
            return data, torch.zeros(len(data))
        num_frames_pred = self.config.sampling.num_frames_pred
        dataset_train, dataset_test = get_dataset( self.config, video_frames_pred=num_frames_pred)
        dataloader = DataLoader(dataset_test, batch_size=self.config.sampling.batch_size // preds_per_test, shuffle=False,
                                num_workers=self.config.data.num_workers, drop_last=False, collate_fn=my_collate)
        vid_mse, vid_ssim, vid_lpips = [], [], []
        Rael_result, predict_result = [], []
        sampler = self.get_sampler()
        # from tmp import img_predictmin
        for i, (real_, _) in tqdm(enumerate(dataloader), total=min(max_data_iter, len(dataloader)),
                                  desc="\nvideo_gen dataloader"):
            if i >= max_data_iter:  # stop early
                break
            real_ = data_transform(self.config, real_)
            logging.info(f"(1) >>>>>Video Pred")
            # Video Prediction
            logging.info( f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")
            real, cond, cond_mask = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred,)
            real = inverse_data_transform(self.config, real)
            cond_original = inverse_data_transform(self.config, cond.clone())
            cond = cond.to(self.config.device)
            # z
            init_samples_shape = (real.shape[0], self.config.data.channels * self.config.data.num_frames,
                                  self.config.data.image_height, self.config.data.image_width)
            init_samples = torch.randn(init_samples_shape, device=self.config.device)
            if getattr(self.config.sampling, 'one_frame_at_a_time', False):
                n_iter_frames = num_frames_pred
            else:
                n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)
            pred_samples = []
            for i_frame in tqdm(range(n_iter_frames), total=n_iter_frames, desc="Generating video frames"):
                mynet = scorenet
                gen_samples = sampler(init_samples,  mynet, cond=cond, cond_mask=cond_mask,
                    n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                    verbose=True if not train else False, final_only=True, denoise=self.config.sampling.denoise,
                    subsample_steps=getattr(self.config.sampling, 'subsample', None),
                    clip_before=getattr(self.config.sampling, 'clip_before', True),
                    t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True if not train else False,
                    gamma=getattr(self.config.model, 'gamma', False))
                gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels * self.config.data.num_frames,
                                                      self.config.data.image_height, self.config.data.image_width)
                # gen_samples = torch.where(gen_samples < 0, 0, gen_samples)
                if config.data.revise:
                    gen_samples = gen_samples.reshape(gen_samples.shape[0], -1, config.data.channels, self.config.data.image_height,self.config.data.image_width).permute(0, 1, 3, 4, 2)
                    gen_samples = torch.squeeze(reshape_patch_back_torch(gen_samples, self.config.data.patch))
                    gen_samples = inverse_data_transform(self.config, gen_samples)
                    gen_samples = model2(gen_samples)
                    gen_samples = data_transform(self.config, gen_samples)
                    gen_samples = gen_samples.unsqueeze(-1)
                    gen_samples = reshape_patch_torch(gen_samples, self.config.data.patch)
                    gen_samples =gen_samples.permute(0, 1, 4, 2, 3)
                    gen_samples = gen_samples.reshape(gen_samples.shape[0], gen_samples.shape[1]*gen_samples.shape[2], gen_samples.shape[3], gen_samples.shape[4])

                pred_samples.append(gen_samples.to('cpu'))
                cond = torch.cat([cond[:, self.config.data.channels * self.config.data.num_frames:],
                                      gen_samples[:, self.config.data.channels * max(0, self.config.data.num_frames - self.config.data.num_frames_cond):]
                                      ], dim=1)
                init_samples = torch.randn(init_samples_shape, device=self.config.device)
            pred = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels * num_frames_pred]
            pred = inverse_data_transform(self.config, pred)
            Rael_result.append(real)
            predict_result.append(pred)

            if real.shape[1] < pred.shape[1]:  # We cannot calculate MSE, PSNR, SSIM
                print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
                for ii in range(len(pred)):
                    vid_mse.append(0)
                    vid_ssim.append(0)
                    vid_lpips.append(0)
            else:
                # Calculate MSE, PSNR, SSIM
                for ii in range(len(pred)):
                    mse, avg_ssim, avg_distance = 0, 0, 0
                    for jj in range(num_frames_pred):

                        # MSE (and PSNR)
                        pred_ij = pred[ii, (self.config.data.channels * jj):(
                                    self.config.data.channels * jj + self.config.data.channels), :, :]
                        real_ij = real[ii, (self.config.data.channels * jj):(
                                    self.config.data.channels * jj + self.config.data.channels), :, :]
                        mse += F.mse_loss(real_ij, pred_ij)

                        # pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                        # real_ij_pil = Transforms.ToPILImage()(real_ij).convert("RGB")

                        # SSIM
                        # pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))
                        # real_ij_np_grey = np.asarray(real_ij_pil.convert('L'))

                        if self.config.data.dataset.upper() == self.config.data.dataset:
                            # ssim is the only metric extremely sensitive to gray being compared to b/w
                            pred_ij_np_grey = np.asarray(
                                Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                            real_ij_np_grey = np.asarray(
                                Transforms.ToPILImage()(torch.round(real_ij)).convert("RGB").convert('L'))
                        # avg_ssim += ssim(pred_ij_np_grey, real_ij_np_grey, data_range=250, gaussian_weights=True,
                        #                  use_sample_covariance=False)
                    vid_mse.append(mse / num_frames_pred)
                    vid_ssim.append(avg_ssim / num_frames_pred)
            if i == 0 or preds_per_test == 1:  # Save first mini-batch or save them all
                cond = cond_original
                data = cond.reshape(cond.shape[0], -1, config.data.channels, self.config.data.image_height,
                                    self.config.data.image_width).permute(0, 1, 3, 4, 2)
                data = np.squeeze(reshape_patch_back(data.numpy(), self.config.data.patch))
                cond = torch.from_numpy(data)

                data = pred.reshape(pred.shape[0], -1, config.data.channels, self.config.data.image_height,
                                    self.config.data.image_width).permute(0, 1, 3, 4, 2)
                data = np.squeeze(reshape_patch_back(data.numpy(), self.config.data.patch))
                pred = torch.from_numpy(data)
                data = real.reshape(real.shape[0], -1, config.data.channels, self.config.data.image_height,
                                    self.config.data.image_width).permute(0, 1, 3, 4, 2)
                data = np.squeeze(reshape_patch_back(data.numpy(), self.config.data.patch))
                real = torch.from_numpy(data)
                gif_frames_cond = []
                gif_frames_pred, gif_frames_pred2, gif_frames_pred3 = [], [], []

                for t in range(cond.shape[1] // self.config.data.outchannels):
                    cond_t = cond[:, t * self.config.data.outchannels:(t + 1) * self.config.data.outchannels]  # BCHW
                    frame = torch.cat([cond_t, 0.5 * torch.ones(*cond_t.shape[:-1], 2), cond_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t + 1:2d}p", (4, 15), 0, 0.5, (1, 1, 1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2 * cond.shape[0]) / 2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6,
                                          pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_cond.append((gif_frame * config.data.Weights).astype('uint8'))
                    if t == 0:
                        gif_frames_cond.append((gif_frame * config.data.Weights).astype('uint8'))
                    del frame, gif_frame
                for t in range(pred.shape[1] // self.config.data.outchannels):
                    real_t = real[:, t * self.config.data.outchannels:(t + 1) * self.config.data.outchannels]  # BCHW
                    pred_t = pred[:, t * self.config.data.outchannels:(t + 1) * self.config.data.outchannels]  # BCHW
                    frame = torch.cat([real_t, 0.5 * torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
                    frame = frame.permute(0, 2, 3, 1).numpy()
                    frame = np.stack([putText(f.copy(), f"{t + 1:02d}", (4, 15), 0, 0.5, (1, 1, 1), 1) for f in frame])
                    nrow = ceil(np.sqrt(2 * pred.shape[0]) / 2)
                    gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6,
                                          pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                    gif_frames_pred.append((gif_frame * config.data.Weights).astype('uint8'))
                    if t == pred.shape[1] // self.config.data.outchannels - 1:
                        gif_frames_pred.append((gif_frame * config.data.Weights).astype('uint8'))
                    del frame, gif_frame

                # Save gif
                output_path = os.path.join(self.args.log_sample_path if train else self.args.video_folder,
                                           f"videos_pred_{ckpt}_{i}.gif")
                # save_pred_gif(gif_frames_cond, gif_frames_pred, output_path)
                del gif_frames_cond, gif_frames_pred, gif_frames_pred2, gif_frames_pred3
                # videos_pred_path = os.path.join(self.args.log_sample_path, f"videos_pred_{ckpt}.pt")
                # video_folder_videos_pred_path = os.path.join(self.args.video_folder, f"videos_pred_{ckpt}.pt")
                # videos_stretch_pred_path = os.path.join(self.args.log_sample_path, f"videos_stretch_pred_{ckpt}_{i}.png")
                # videos_stretch_pred_test_path = os.path.join(self.args.video_folder, f"videos_stretch_pred_{ckpt}_{i}.png")
                # save_pred(pred, real, train, cond, config, ckpt, videos_pred_path, video_folder_videos_pred_path,
                #           videos_stretch_pred_path, videos_stretch_pred_test_path)
                # save_pred(pred, real)
        Rael_result = torch.cat(Rael_result, dim=0)
        predict_result = torch.cat(predict_result, dim=0)
        torch.save(Rael_result, os.path.join(self.args.video_folder, f"Rael_result.pt"))
        torch.save(predict_result, os.path.join(self.args.video_folder, f"Predict_result.pt"))


        # Calc MSE, PSNR, SSIM, LPIPS
        mse_list = np.array(vid_mse).reshape(-1, preds_per_test).min(-1)
        psnr_list = (10 * np.log10(1 / np.array(vid_mse))).reshape(-1, preds_per_test).max(-1)
        ssim_list = np.array(vid_ssim).reshape(-1, preds_per_test).max(-1)
        lpips_list = np.array(vid_lpips).reshape(-1, preds_per_test).min(-1)

        def image_metric_stuff(metric):
            avg_metric, std_metric = metric.mean().item(), metric.std().item()
            conf95_metric = avg_metric - float(st.norm.interval(alpha=0.95, loc=avg_metric, scale=st.sem(metric))[0])
            return avg_metric, std_metric, conf95_metric

        avg_mse, std_mse, conf95_mse = image_metric_stuff(mse_list)
        avg_psnr, std_psnr, conf95_psnr = image_metric_stuff(psnr_list)
        avg_ssim, std_ssim, conf95_ssim = image_metric_stuff(ssim_list)
        avg_lpips, std_lpips, conf95_lpips = image_metric_stuff(lpips_list)

        vid_metrics = {'ckpt': ckpt, 'preds_per_test': preds_per_test,
                       'mse': avg_mse, 'mse_std': std_mse, 'mse_conf95': conf95_mse,
                       'psnr': avg_psnr, 'psnr_std': std_psnr, 'psnr_conf95': conf95_psnr,
                       'ssim': avg_ssim, 'ssim_std': std_ssim, 'ssim_conf95': conf95_ssim,
                       'lpips': avg_lpips, 'lpips_std': std_lpips, 'lpips_conf95': conf95_lpips}


        elapsed = str(datetime.timedelta(seconds=(time.time() - self.start_time)))[:-3]
        format_p = lambda dd: ", ".join([f"{k}:{v:.4f}" if k != 'ckpt' and k != 'preds_per_test' and k != 'time' else f"{k}:{v:7d}" if k == 'ckpt' else f"{k}:{v:3d}" if k == 'preds_per_test' else f"{k}:{v}"
                                            for k, v in dd.items()])
        logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
        logging.info(f"elapsed: {elapsed}, mem:{get_proc_mem():.03f}GB, GPUmem: {get_GPU_mem():.03f}GB")
        #
        if train:
            return vid_metrics
        else:
            logging.info(
                f"elapsed: {elapsed}, Writing metrics to {os.path.join(self.args.video_folder, 'vid_metrics.yml')}")
            vid_metrics['time'] = elapsed
            if self.condp == 0.0 and self.futrf == 0:  # (1) Prediction
                vid_metrics['pred_mse'], vid_metrics['pred_psnr'], vid_metrics['pred_ssim'], vid_metrics['pred_lpips'] = \
                vid_metrics['mse'], vid_metrics['psnr'], vid_metrics['ssim'], vid_metrics['lpips']
                vid_metrics['pred_mse_std'], vid_metrics['pred_psnr_std'], vid_metrics['pred_ssim_std'], vid_metrics[
                    'pred_lpips_std'] = vid_metrics['mse_std'], vid_metrics['psnr_std'], vid_metrics['ssim_std'], \
                                        vid_metrics['lpips_std']
                vid_metrics['pred_mse_conf95'], vid_metrics['pred_psnr_conf95'], vid_metrics['pred_ssim_conf95'], \
                vid_metrics['pred_lpips_conf95'] = vid_metrics['mse_conf95'], vid_metrics['psnr_conf95'], vid_metrics[
                    'ssim_conf95'], vid_metrics['lpips_conf95']
            logging.info(f"elapsed: {elapsed}, {format_p(vid_metrics)}")
            self.write_to_yaml(os.path.join(self.args.video_folder, 'vid_metrics.yml'), vid_metrics)
    #
    def valid(self):
        # model = AttU_Net(5, 5).cuda()
        # model.load_state_dict(torch.load('/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_2/video_samples/unet-v5_118_6_weight/generator_r78.pth', map_location=self.config.device))
        # model.eval()
        self.start_time = time.time()
        preds_per_test = getattr(self.config.sampling, 'preds_per_test', 1)
        # Conditional
        conditional = self.config.data.num_frames_cond > 0
        future = getattr(self.config.data, "num_frames_future", 0)
        ckpt = self.config.sampling.ckpt_id
        ckpt_file = os.path.join(self.args.log_path, f'checkpoint.pt')
        logging.info(f"Loading ckpt {ckpt_file}")
        states = torch.load(ckpt_file, map_location=self.config.device)
        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        scorenet.load_state_dict(states[0], strict=False)
        scorenet.eval()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)

        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Collate fn for n repeats
        def my_collate(batch):
            data, _ = zip(*batch)
            data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
            return data, torch.zeros(len(data))
        num_frames_pred = self.config.sampling.num_frames_pred
        dataset, _ = get_dataset(self.config, video_frames_pred=num_frames_pred, valid=True)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size // preds_per_test, shuffle=False,
                                num_workers=16, drop_last=False, collate_fn=my_collate)
        sampler = self.get_sampler()
        result = []
        for i, (real_, _) in tqdm(enumerate(dataloader),desc="\nvideo_gen dataloader"):

            real_ = data_transform(self.config, real_)
            logging.info(f"(1) >>>>>Video 'Pred' ")
            if future == 0:
                num_frames_pred = self.config.sampling.num_frames_pred
                logging.info(
                    f"PREDICTING {num_frames_pred} frames, using a {self.config.data.num_frames} frame model conditioned on {self.config.data.num_frames_cond} frames, subsample={getattr(self.config.sampling, 'subsample', None)}, preds_per_test={preds_per_test}")
            _, cond, _ = conditioning_fn(self.config, real_, num_frames_pred=num_frames_pred, conditional=conditional)

            cond = cond.to(self.config.device)
            init_samples_shape = (cond.shape[0], self.config.data.channels * self.config.data.num_frames,
                                  self.config.data.image_height, self.config.data.image_width)
            init_samples = torch.randn(init_samples_shape, device=self.config.device)
            n_iter_frames = ceil(num_frames_pred / self.config.data.num_frames)
            pred_samples = []
            print('cond', cond.shape, 'init_samples', init_samples.shape)
            for i_frame in tqdm(range(n_iter_frames), total=n_iter_frames, desc="Generating video frames"):
                mynet = scorenet

                # Generate samples
                gen_samples = sampler(
                    init_samples if i_frame == 0 or getattr(self.config.sampling, 'init_prev_t',-1) <= 0 else gen_samples,
                    mynet, cond=cond, cond_mask=None,
                    n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr,
                    verbose=True , final_only=True, denoise=self.config.sampling.denoise,
                    subsample_steps=getattr(self.config.sampling, 'subsample', None),
                    clip_before=getattr(self.config.sampling, 'clip_before', True),
                    t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True,
                    gamma=getattr(self.config.model, 'gamma', False))

                print('gen_samples', gen_samples.shape)
                gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0],
                                                      self.config.data.channels * self.config.data.num_frames,
                                                      self.config.data.image_height, self.config.data.image_width)
                if config.data.revise:
                    gen_samples = gen_samples.reshape(gen_samples.shape[0], -1, config.data.channels, self.config.data.image_height,self.config.data.image_width).permute(0, 1, 3, 4, 2)
                    gen_samples = torch.squeeze(reshape_patch_back_torch(gen_samples, self.config.data.patch))
                    gen_samples = inverse_data_transform(self.config, gen_samples)
                    # gen_samples = model(gen_samples)
                    gen_samples = data_transform(self.config, gen_samples)
                    gen_samples = gen_samples.unsqueeze(-1)
                    gen_samples = reshape_patch_torch(gen_samples, self.config.data.patch)
                    gen_samples =gen_samples.permute(0, 1, 4, 2, 3)
                    gen_samples = gen_samples.reshape(gen_samples.shape[0], gen_samples.shape[1]*gen_samples.shape[2], gen_samples.shape[3], gen_samples.shape[4])

                pred_samples.append(gen_samples.to('cpu'))
                cond = torch.cat([cond[:, self.config.data.channels * self.config.data.num_frames:],
                                  gen_samples[:, self.config.data.channels * max(0, self.config.data.num_frames - self.config.data.num_frames_cond):]
                                  ], dim=1)
            pred = torch.cat(pred_samples, dim=1)[:, :self.config.data.channels * num_frames_pred]
            pred = inverse_data_transform(self.config, pred)
            # pred = torch.where(pred < 0, 0, pred)
            data = pred.reshape(pred.shape[0], -1, config.data.channels, self.config.data.image_height,
                                self.config.data.image_width).permute(0, 1, 3, 4, 2)
            data = np.squeeze(reshape_patch_back(data.detach().numpy(), self.config.data.patch))
            pred = torch.from_numpy(data)
            # gif_frames_pred = []
            # for t in range(pred.shape[1] // self.config.data.outchannels):
            #     pred_t = pred[:, t * self.config.data.outchannels:(t + 1) * self.config.data.outchannels]  # BCHW
            #     real_t = torch.zeros_like(pred_t)
            #     frame = torch.cat([real_t, 0.5 * torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
            #     frame = frame.permute(0, 2, 3, 1).numpy()
            #     frame = np.stack([putText(f.copy(), f"{t + 1:02d}", (4, 15), 0, 0.5, (1, 1, 1), 1) for f in frame])
            #     nrow = ceil(np.sqrt(2 * pred.shape[0]) / 2)
            #     gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6,
            #                           pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
            #     gif_frames_pred.append((gif_frame * config.data.Weights).astype('uint8'))
            #     if t == pred.shape[1] // self.config.data.outchannels - 1 and future == 0:
            #         gif_frames_pred.append((gif_frame * config.data.Weights).astype('uint8'))
            #     del frame, gif_frame
            #
            # # Save gif
            # if self.condp == 0.0 and self.futrf == 0:  # (1) Prediction
            #     imageio.mimwrite(os.path.join(self.args.video_folder, f"videos_pred_{ckpt}_{i}.gif"), [ *gif_frames_pred], fps=4)
            result.append(pred)
        result = torch.cat(result, dim=0)
        print(result.shape)
        torch.save(result, os.path.join(self.args.video_folder, f"resultssss.pt"))
        # imgwrite_(result, config)




if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    args, config, config_uncond = parse_args_and_config()
    # os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    # args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)

    # print(args.image_folder)
    version = getattr(config.model, 'version', "SMLD")
    Runner = NCSNRunner(args, config, config_uncond)

    if args.train:
        Runner.train()
    # elif args.sample:
    #     Runner.sample()
    elif args.video_gen:
        Runner.video_gen()
    else:
        Runner.valid()


    # CUDA_VISIBLE_DEVICES=5 python runner.py --config configs/tianchi.yml --exp radar_ch_4_size_128_num --config_mod sampling.subsample=100

