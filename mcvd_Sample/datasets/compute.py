from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import math

# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms


def thresholdsloss(pred, label, inx):

    weights = np.ones_like(pred)
    for i, threshold in enumerate([10, 30, 40, 50, 70]):
        w = np.where(label >= threshold, 1, 0.0)
        weights = weights + w
    mse = np.mean(weights * ((pred - label) ** 2))

    return mse

def calc_metrics(A, B, inx):

    # 计算 SSIM
    score, _ = compare_ssim(A, B, full=True,)


    # 计算 MSE
    # mse_score = np.mean((A - B) ** 2)
    mse_score = thresholdsloss(A, B, inx)
    mse = np.mean((B - A) ** 2)
    max_pixel_value = 70
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)

    return score, mse_score, psnr
