import preprocessing
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error

def denormalize_images(normalized_images, params_dict, filenames):
    restored_images = []
    for i, filename in enumerate(filenames):
        channel_params = params_dict[filename]
        norm_img = normalized_images[i]
        restored = np.zeros_like(norm_img)

        for ch in range(3):
            scale = channel_params[ch][2]
            offset = channel_params[ch][3]
            restored[..., ch] = (norm_img[..., ch] - offset) / scale

        restored_images.append(restored)
    return np.array(restored_images)


def evaluate_generator(g_model, weighted_dir, map_dir):

    [X_weighted, X_map], (params_w, params_map) = preprocessing.load_real_samples(weighted_dir, map_dir)
    filenames = sorted(list(set(os.listdir(weighted_dir)) & set(os.listdir(map_dir))))

    X_fake_map = g_model.predict(X_weighted, verbose=1)

    X_fake_map_denorm = denormalize_images(X_fake_map, params_map, filenames)
    X_map_denorm = denormalize_images(X_map, params_map, filenames)

    ssim_values = [[], [], []]
    psnr_values = [[], [], []]

    n = len(filenames)

    for i in range(n):
        true_img = X_map_denorm[i]
        pred_img = X_fake_map_denorm[i]

        for ch in range(3):
            t = true_img[:, :, ch]
            p = pred_img[:, :, ch]

            data_range = t.max() - t.min() if t.max() != t.min() else 1.0

            ssim_val = ssim(t, p, data_range=data_range)
            psnr_val = psnr(t, p, data_range=data_range)

            ssim_values[ch].append(ssim_val)
            psnr_values[ch].append(psnr_val)

    channel_names = ['T1', 'T2', 'PD']
    for ch in range(3):
        avg_ssim = np.mean(ssim_values[ch])
        std_ssim = np.std(ssim_values[ch])

        avg_psnr = np.mean(psnr_values[ch])
        std_psnr = np.std(psnr_values[ch])

        print(f"\n{channel_names[ch]}:")
        print(f"  SSIM: {avg_ssim:.5f} ± {std_ssim:.5f}")
        print(f"  PSNR: {avg_psnr:.5f} ± {std_psnr:.5f} dB")

    return