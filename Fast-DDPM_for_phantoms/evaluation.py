import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity

PATH_FOR_IMG = 'exp/image_samples/Fast-DDPM_experiments/images_fid'

pt = np.load(f'{PATH_FOR_IMG}/subject18_crisp_v_180_350000_pt_np.npy')

gt = np.load(f'{PATH_FOR_IMG}/subject18_crisp_v_180_350000_gt_np.npy')


plt.figure()

f, axs = plt.subplots(1,3)

axs[0].imshow(gt[0, :, :])

axs[1].imshow(gt[1, :, :])

axs[2].imshow(gt[2, :, :])

plt.show()


plt.figure()

f, axs = plt.subplots(1,3)

axs[0].imshow(pt[0, :, :])

axs[1].imshow(pt[1, :, :])

axs[2].imshow(pt[2, :, :])

plt.show()

range_list = list(range(100, 200))

def calculate_metrics_diffusion(range_list):
    ssim_list = []
    psnr_list = []
    #mse_list = []
    nmse_list = []

    ssim_T1 = []
    ssim_T2 = []
    ssim_PD = []

    psnr_T1 = []
    psnr_T2 = []
    psnr_pd = []

    for i in range_list:
        gt = np.load(f'{PATH_FOR_IMG}/subject18_crisp_v_{i}_350000_gt_np.npy')
        pt = np.load(f'{PATH_FOR_IMG}/subject18_crisp_v_{i}_350000_pt_np.npy')


        for j in range(gt.shape[0]):

            data_range = pt[j, :, :].max() - abs(pt[j, :, :].min())
            (score_ssim, diff) = structural_similarity(gt[j, :, :], pt[j, :, :], data_range = data_range, full=True)
            ssim_list.append(score_ssim)

            normalized_gt = cv2.normalize(gt[j, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normalized_pt = cv2.normalize(pt[j, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            psnr = cv2.PSNR(normalized_gt, normalized_pt)
            psnr_list.append(psnr)

            mse = np.mean((gt[j, :, :] - pt[j, :, :]) ** 2)
            nmse = mse/np.var(gt[j, :, :])
            nmse_list.append(nmse)

            if j == 0:
                ssim_T1.append(score_ssim)
                psnr_T1.append(psnr)
            if j == 1:
                ssim_T2.append(score_ssim)
                psnr_T2.append(psnr)
            if j == 2:
                ssim_PD.append(score_ssim)
                psnr_pd.append(psnr)


    print('All_img_ssim: ', np.mean(ssim_list),
          'All_img_psnr: ', np.mean(psnr_list),
          'All_img_NMSE: ', np.mean(nmse_list),
          'T1 ssim: ', np.mean(ssim_T1),
          'T2 ssim: ', np.mean(ssim_T2),
          'PD ssim: ', np.mean(ssim_PD),
          'PSNR T1: ', np.mean(psnr_T1),
          'PSNR T2: ', np.mean(psnr_T2),
          'PSNR PD: ', np.mean(psnr_pd)
    )

    return ssim_list, psnr_list

calculate_metrics_diffusion(range_list)
