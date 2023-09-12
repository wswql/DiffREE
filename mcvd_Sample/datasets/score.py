import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from configs.tools import reshape_patch_back,reshape_patch
import os
import cv2
import imageio
from tqdm import tqdm
from PIL import Image
import os.path
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def imgwrite_( data, image_height=64, image_width=128,   channels=4, ):
    datapath = '/data03/wuqiliang/code/mcvd_Sample/videos/result.pt'
    resultpath = '/data03/wuqiliang/code/mcvd_Sample/data/radar'
    namelist = np.load(os.path.join(resultpath, 'index.npy'))
    data = torch.load(datapath)
    print(data.shape)
    data = data.reshape(data.shape[0], -1, channels, 64, 128).permute(0, 1, 3, 4, 2)
    print(data.shape)
    # real_gif = []
    data = np.squeeze(reshape_patch_back(data.numpy(), 2))
    print(data.shape)
    # for dir in range(len(namelist)):
    #     dirpath = os.path.join(resultpath, 'result/' + str(int(namelist[dir])))
    #     os.makedirs(dirpath, exist_ok=True)
    #     for i in range(data.shape[1]):
    #         png = data[dir][i] * 250
    #         real_gif.append(png.astype('uint8'))
    #         print(os.path.join(dirpath,'pre_'+str(int(namelist[dir]))+'_'+str(i+13)+'.png'))
    #         cv2.imwrite(os.path.join(dirpath,'pre_'+str(int(namelist[dir]))+'_'+str(i+13)+'.png'), png)
        # imageio.mimwrite(f"/data03/wuqiliang/code/mcvd_Sample/videos/result.gif", real_gif, fps=4)
    #     # print(data.shape)
# imgwrite_(None)




def score_SSIM_PSNR():

    data = torch.load('/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_1/video_samples/Predict_result.pt')
    data = data.reshape(data.shape[0], -1, 4, 128, 128).permute(0, 1, 3, 4, 2)
    matrix1 = np.squeeze(reshape_patch_back(data.numpy(), 2))

    data = torch.load('/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_1/video_samples/Rael_result.pt')
    data = data.reshape(data.shape[0], -1, 4, 128, 128).permute(0, 1, 3, 4, 2)
    matrix2 = np.squeeze(reshape_patch_back(data.numpy(), 2))
    (batchsize, num_channels, h, w) = matrix2.shape

    total_ssim = 0
    total_psnr = 0

    for i in range(batchsize):
        for j in range(num_channels):
            channel_ssim = ssim(matrix1[i, j], matrix2[i, j])
            channel_psnr = psnr(matrix1[i, j], matrix2[i, j])
            total_ssim += channel_ssim
            total_psnr += channel_psnr

    average_ssim = total_ssim / (batchsize * num_channels)
    average_psnr = total_psnr / (batchsize * num_channels)

    print(f"Average SSIM: {average_ssim}")
    print(f"Average PSNR: {average_psnr}")






def _ETS_calc(nas, nbs, ncs, nds):
    e = (nas + ncs) * (nas + nbs) / (nas + nbs + ncs + nds)
    ETS = (nas - e) / (nas + nbs + ncs - e)
    return ETS


def _HSS_calc(nas, nbs, ncs, nds):
    HSS = (2 * (nas * nds - ncs * nbs)) / ((nas + ncs) * (ncs + nds) + (nas + nbs) * (nbs + nds))
    return HSS


def _CSI_calc(nas, nbs, ncs, nds):
    TS = nas / (nas + nbs + ncs)
    return TS


def _POD_calc(nas, nbs, ncs, nds):
    POD = nas / (nas + ncs)
    return POD


# Define the ground truth and predicted matrices (replace these with your actual data)


def get_score_2d(pred:np.ndarray, label:np.ndarray):
    THD = [10, 20, 30, 50]
    NARN, NBRN, NCRN, NDRN, = 0, 0, 0, 0,
    len = 0
    for k in range(2):
        tmp_label2D = np.copy(label)
        tmp_label2D[tmp_label2D > THD[k+1]] = 0
        l2D = np.where(tmp_label2D >= THD[k], 2.0, 0.0)
        N = np.sum(l2D == 2)
        tmp_pred2D = np.copy(pred)
        tmp_pred2D[tmp_pred2D > THD[k+1]] = 0
        p2D = np.where(tmp_pred2D >= THD[k], 1.0, 0.0)
        r2D = l2D - p2D
        NA = np.sum(r2D == 1).astype(np.float32)   # 命中
        NB = np.sum(r2D == -1).astype(np.float32)  # 空报
        NC = np.sum(r2D == 2).astype(np.float32)   # 漏报
        ND = np.sum(r2D == 0).astype(np.float32)   # 均无
        NARN += NA
        NBRN += NB
        NCRN += NC
        NDRN += ND
        len += N
    # score = _SCORE_calc(NARN, NBRN, NCRN, NDRN, len)
    CSI = _CSI_calc(NARN, NBRN, NCRN, NDRN)
    ETS = _ETS_calc(NARN, NBRN, NCRN, NDRN)
    POD = _POD_calc(NARN, NBRN, NCRN, NDRN)
    HSS = _HSS_calc(NARN, NBRN, NCRN, NDRN)

    return HSS, ETS, CSI, POD



def calculate_metrics(ground_truth, predicted):
    ETS_values = []
    CSI_values = []
    POD_values = []
    ssim_values = []
    mse_values = []
    HSS_values = []
    for i in range(batchsize):
        for j in range(num_channels):
            gt_channel = ground_truth[i, j]
            pred_channel = predicted[i, j]
            HSS, ETS, CSI, POD = get_score_2d(gt_channel, pred_channel)

            ssim_score = ssim(gt_channel, pred_channel)
            mse_score = mean_squared_error(gt_channel/70, pred_channel/70)
            ETS_values.append(round(ETS, 2))
            CSI_values.append(round(CSI, 2))
            POD_values.append(round(POD, 2))
            ssim_values.append(round(ssim_score, 2))
            mse_values.append(round(mse_score, 2))
            HSS_values.append(round(HSS, 2))
    ETS_values = np.round(np.nanmean(np.array(ETS_values).reshape(batchsize, num_channels), axis=0), 2)
    CSI_values = np.round(np.nanmean(np.array(CSI_values).reshape(batchsize, num_channels), axis=0), 2)
    POD_values = np.round(np.nanmean(np.array(POD_values).reshape(batchsize, num_channels), axis=0), 2)
    HSS_values = np.round(np.nanmean(np.array(HSS_values).reshape(batchsize, num_channels), axis=0), 2)
    ssim_values = np.round(np.nanmean(np.array(ssim_values).reshape(batchsize, num_channels), axis=0), 2)
    mse_values = np.round(np.nanmean(np.array(mse_values).reshape(batchsize, num_channels), axis=0), 2)
    pltimge(ETS_values, CSI_values, POD_values, ssim_values, mse_values, HSS_values)
    return (ETS_values, CSI_values, POD_values, ssim_values, mse_values, HSS_values)







def pltimge(average_ETS, average_CSI, average_POD, average_ssim, average_mse, average_HSS):
    # Assuming you have calculated the average metrics for each channel
    # average_ETS, average_CSI, average_POD, average_ssim, average_mse, average_HSS
    metrics_names = ['ETS', 'CSI', 'POD', 'SSIM', 'MSE', 'HSS']
    average_metrics = [average_ETS, average_CSI, average_POD, average_ssim, average_mse, average_HSS]
    # Define colors for each metric
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    for i, metric_values in enumerate(average_metrics):
        plt.plot(metric_values, label=metrics_names[i], color=colors[i], marker='o')
    plt.xlabel('Channels')
    plt.ylabel('Average Value')
    plt.title('Average Evaluation Metrics Across Channels')
    plt.xticks(range(len(metric_values)))  # Assuming all metrics have the same length
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    score_SSIM_PSNR()
    # data = torch.load('/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_num_1/video_samples/Pre_Predict_result.pt') * 70
    # data = data.reshape(data.shape[0], -1, 4, 128, 128).permute(0, 1, 3, 4, 2)
    # Pre_predicted = np.squeeze(reshape_patch_back(data.numpy(), 2))

    #
    data = torch.load(
        '/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_1/video_samples/Predict_result.pt') * 70
    data = data.reshape(data.shape[0], -1, 4, 128, 128).permute(0, 1, 3, 4, 2)
    predicted = np.squeeze(reshape_patch_back(data.numpy(), 2))
    #
    data = torch.load('/data03/wuqiliang/code/mcvd_Sample/radar_ch_4_size_128_tianchi_1/video_samples/Rael_result.pt') * 70
    data = data.reshape(data.shape[0], -1, 4, 128, 128).permute(0, 1, 3, 4, 2)
    ground_truth = np.squeeze(reshape_patch_back(data.numpy(), 2))

    (batchsize, num_channels, h, w) = ground_truth.shape

    # score_SSIM_PSNR()
    ETS_values, CSI_values, POD_values, ssim_values, mse_values, HSS_values = calculate_metrics(ground_truth,predicted)
    print("Average ETS:", ETS_values)
    print("Average CSI:", CSI_values)
    print("Average POD:", POD_values)
    print("Average SSIM:", ssim_values)
    print("Average MSE:", mse_values)
    print("Average HSS:", HSS_values)
    # ETS_values, CSI_values, POD_values, ssim_values, mse_values, HSS_values = calculate_metrics(ground_truth,Pre_predicted)
    # print("Average ETS:", ETS_values)
    # print("Average CSI:", CSI_values)
    # print("Average POD:", POD_values)
    # print("Average SSIM:", ssim_values)
    # print("Average MSE:", mse_values)
    # print("Average HSS:", HSS_values)
