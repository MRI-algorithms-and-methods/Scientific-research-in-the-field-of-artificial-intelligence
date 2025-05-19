import os
import math
import json
import shutil
import filecmp
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def get_dataset_paths(base_dir: str, split: str, contrast: str) -> dict:
    root = Path(base_dir) / split / contrast
    return {
        "phantoms": root / "phantoms",
        "ps": {
            "ks_order": root / "ps" / "ks_order",
            "waveforms": root / "ps" / "waveforms",
            "parameters": root / "ps" / "parameters",
            "seq": root / "ps" / "seq",
        },
        "kspace": {
            "raw": root / "kspace" / "raw",
            "sorted": root / "kspace" / "sorted",
        },
        "recon": {
            "images": root / "recon" / "images",
            "npy": root / "recon" / "npy",
        }
    }




def generate_phantom_ps_dataset(
        contrast_type: str,
        paths: dict,
        phantom_src_dir: str,
        seq_src_dir_template: str,
        param_src_dir_template: str,
        phantom_stride: int = 20,
        sequence_stride: int = 10
):
    """
    Processes and copies phantom, sequence, k-space order, and parameter files from source directories
    to the appropriate output directories. This function ensures that copied files match the originals
    using full file comparison and provides a progress log for each copied file.

    Args:
    - contrast_type (str): The contrast type (e.g., 'T1', 'T2').
    - paths (dict): A dictionary containing the output directories for phantoms, sequences, k-space orders, and parameters.
    - phantom_src_dir (str): Path to the source directory containing phantom image files.
    - seq_src_dir_template (str): Path template for the source sequence directory, formatted with the contrast type.
    - param_src_dir_template (str): Path template for the source parameter directory, formatted with the contrast type.
    - phantom_stride (int, optional): The stride for phantom files (default is 20).
    - sequence_stride (int, optional): The stride for sequence and k-space order files (default is 10).

    Raises:
    - AssertionError: If the count of .seq, .json (k-space order), and .json (parameter) files do not match.
    - AssertionError: If the copied files do not match the originals during the comparison.
    """

    # === Prepare paths ===
    seq_src_dir = seq_src_dir_template.format(contrast_type=contrast_type)
    param_src_dir = param_src_dir_template.format(contrast_type=contrast_type)

    # === Load file lists ===
    phantom_files = os.listdir(phantom_src_dir)[::phantom_stride]
    seq_files = [f for f in os.listdir(seq_src_dir) if f.endswith(".seq")][::sequence_stride]
    ks_order_files = [f for f in os.listdir(seq_src_dir) if f.endswith(".json")][::sequence_stride]
    param_files = [f for f in os.listdir(param_src_dir) if f.endswith(".json")][::sequence_stride]

    assert len(seq_files) == len(ks_order_files) == len(param_files), \
        "Mismatch in .seq, .json (ks_order), and param files count."

    # === Output paths from dictionary ===
    phantoms_path = paths["phantoms"]
    ps_seq_path = paths["ps"]["seq"]
    ps_ks_order_path = paths["ps"]["ks_order"]
    ps_parameters_path = paths["ps"]["parameters"]

    # === Ensure output directories exist ===
    for p in [phantoms_path, ps_seq_path, ps_ks_order_path, ps_parameters_path]:
        os.makedirs(p, exist_ok=True)

    # === Processing loop ===
    counter = 0
    for i, phantom_file in enumerate(phantom_files):
        phantom_path = os.path.join(phantom_src_dir, phantom_file)

        for j in range(len(seq_files)):
            seq_file = os.path.join(seq_src_dir, seq_files[j])
            ks_order_file = os.path.join(seq_src_dir, ks_order_files[j])
            param_file = os.path.join(param_src_dir, param_files[j])

            base_name = f"{phantom_file.replace('.h5', '')}_{seq_files[j].replace('.seq', '')}"

            # Destination file paths
            dst_phantom = os.path.join(phantoms_path, f"{base_name}.h5")
            dst_seq = os.path.join(ps_seq_path, f"{base_name}.seq")
            dst_ks_order = os.path.join(ps_ks_order_path, f"{base_name}_ksof.json")
            dst_param = os.path.join(ps_parameters_path, f"{base_name}.json")

            # Copy and verify files
            shutil.copy2(phantom_path, dst_phantom)
            shutil.copy2(seq_file, dst_seq)
            shutil.copy2(ks_order_file, dst_ks_order)
            shutil.copy2(param_file, dst_param)

            # Assertions for file integrity
            # Assertions for file integrity
            assert (phantom_path.split('subject')[1].split('.h5')[0] == dst_phantom.split('subject')[1].split('TSE')[0][
                                                                        :-1]), \
                f"Phantom file mismatch: {phantom_path} != {dst_phantom}"
            assert (seq_file.split('TSE')[1].split('.seq')[0] , dst_phantom.split('TSE')[1].split('.h5')[0])

            assert (seq_file.split('TSE')[1] == dst_seq.split('TSE')[1]), \
                f"Sequence file mismatch: {seq_files[j]} != {dst_seq}"
            assert (ks_order_file.split('TSE')[1].split('k_space_order_filing')[0] ==
                    dst_ks_order.split('TSE')[1].split('ksof')[0]), \
                f"KSOF file mismatch: {ks_order_file} != {dst_ks_order}"
            assert (param_file.split('TSE')[1] == dst_param.split('TSE')[1]), \
                f"Param file mismatch: {param_file} != {dst_param}"


            # assert (phantom_path.split('subject')[1].split('.h5')[0] == dst_phantom.split('subject')[1].split('TSE')[0][:-1])
            # assert (seq_file.split('TSE')[1].split('.seq')[0] , dst_phantom.split('TSE')[1].split('.h5')[0])
            # assert (seq_file.split('TSE')[1] == dst_seq.split('TSE')[1]), f"Sequence file mismatch: {seq_files[j]}"
            # assert (ks_order_file.split('TSE')[1].split('k_space_order_filing')[0] == dst_ks_order.split('TSE')[1].split('ksof')[0]), f"KSOF file mismatch: {ks_order_files[j]} ! = {dst_ks_order}"
            # assert (param_file.split('TSE')[1] == dst_param.split('TSE')[1]), f"Param file mismatch: {param_files[j]} ! = {dst_param}"
            print(f"[{counter}] Saved: {base_name}")
            counter += 1

class MRIImageDataset(Dataset):
    def __init__(self, ps_dir, phantom_dir, ks_dir, recon_dir, transform=None, norm_mode='minus_one_one', max_ps_length=11024):
        self.ps_dir = ps_dir
        self.phantom_dir = phantom_dir
        self.ks_dir = ks_dir
        self.recon_dir = recon_dir  # Directory containing recon_images
        self.transform = transform
        self.norm_mode = norm_mode  # Normalization mode
        # Determine the maximum sequence length for padding
        # self.max_ps_length = get_max_ps_length(ps_dir)

        self.ps_files = sorted(os.listdir(ps_dir))
        self.phantom_files = sorted(os.listdir(phantom_dir))
        self.ks_files = sorted(os.listdir(ks_dir))
        self.recon_files = sorted(os.listdir(recon_dir))  # List of recon image files
        print(f"The number of files in the directories match. {len(self.ps_files)}, {len(self.phantom_files)} ,  {len(self.ks_files)}, {len(self.recon_files)}.")

        assert len(self.ps_files) == len(self.phantom_files) == len(self.ks_files) == len(self.recon_files), \
            f"Mismatch in the number of files in the directories. {len(self.ps_files)}, {len(self.phantom_files)} ,  {len(self.ks_files)}, {len(self.recon_files)}."

        # Verify file name matching and find max PS length
        self.max_ps_length = max_ps_length
        for ps_file, ph_file, ks_file, recon_file in zip(self.ps_files, self.phantom_files, self.ks_files,
                                                         self.recon_files):
            ps_id = ps_file.split('.npy')[0]
            ph_id = ph_file.split('.h5')[0]
            ks_id = ks_file.split('.npy')[0]
            recon_id = recon_file.split('.npy')[0][2:]  # Assuming recon images are in .h5 format

            assert ps_id == ph_id == ks_id == recon_id, f"Mismatch: {ps_file}, {ph_file}, {ks_file}, {recon_file}"



    def process_ps_data(self, ps_data):
        # Pad PS data with zeros if needed
        if ps_data.shape[1] < self.max_ps_length:
            pad_width = ((0, 0), (0, self.max_ps_length - ps_data.shape[1]))
            ps_data = np.pad(ps_data, pad_width, mode='constant')

        # Normalize each row of ps_data differently
        # ps_data[0, :] - leave it as is
        ps_data[0, :] = ps_data[0, :]  # No change (you can skip this step as it does nothing)

        # ps_data[4, :] - Normalize to [-1, 1]
        ps_data[4, :] = normalize_data(ps_data[4, :], mode='minus_one_one')

        # All other rows - Normalize to [0, 1]
        for i in range(ps_data.shape[0]):
            if i != 0 and i != 4:  # Skip 0 and 4, which are handled separately
                ps_data[i, :] = normalize_data(ps_data[i, :], mode='zero_one')

        return ps_data

    def __len__(self):
        return len(self.ps_files)

    def __getitem__(self, idx):
        ps_path = os.path.join(self.ps_dir, self.ps_files[idx])
        ps_data = np.load(ps_path).T

        # Process the PS data (padding and normalization)
        ps_data = self.process_ps_data(ps_data)

        phantom_path = os.path.join(self.phantom_dir, self.phantom_files[idx])
        phantom_data = read_h5(phantom_path)


        ks_path = os.path.join(self.ks_dir, self.ks_files[idx])
        ks_data =  np.load(ks_path)

        # print(ps_data.shape, phantom_data.shape, ks_data.shape)


        recon_path = os.path.join(self.recon_dir, self.recon_files[idx])
        recon_data = np.load(recon_path)  # Assuming recon images are stored in .h5 format
        recon_data = normalize_data(recon_data, self.norm_mode)

        ks_real = np.abs(ks_data)
        ks_real = normalize_data(ks_real, self.norm_mode)
        ks_imag = np.angle(ks_data) / np.pi



        # Normalize each channel of phantom_data separately
        if phantom_data.ndim == 3:  # Assuming phantom_data has shape (channels, rows, columns)
            for c in range(phantom_data.shape[-1]):  # Iterate through channels
                phantom_data[:, :, c] = normalize_data(phantom_data[:, :, c], self.norm_mode)

        # Apply the transformation if provided (default is ToTensor + Normalize)
        ks_real = self.transform(ks_real)
        ks_imag = self.transform(ks_imag)
        recon_data = self.transform(recon_data)  # Apply transformation to recon image

        ps_tensor = torch.tensor(ps_data, dtype=torch.float32)
        phantom_tensor = torch.tensor(phantom_data, dtype=torch.float32)
        ks_real_tensor = torch.tensor(ks_real, dtype=torch.float32)
        ks_imag_tensor = torch.tensor(ks_imag, dtype=torch.float32)
        recon_tensor = torch.tensor(recon_data, dtype=torch.float32)  # Convert recon image to tensor
        # Return additional recon_tensor along with other data
        return (ps_tensor, phantom_tensor), (ks_real_tensor, ks_imag_tensor)

        # return (ps_tensor, phantom_tensor), recon_tensor




def read_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        key = list(f.keys())[0]
        data = f[key]
        keyw = list(data.keys())[0]
        return np.array(data[keyw])

# Plotting functions
def plot_ps(ps_array):
    for i in range(ps_array.shape[0]):
        plt.subplot(ps_array.shape[0], 1, i + 1)
        plt.plot(ps_array[i, :])
    plt.show()

def plot_phantom(phantom_data):
    for ph in range(5):
        slice_np = phantom_data[:, :, ph].cpu().numpy()  # Add .cpu() if it's on GPU

        plt.subplot(2,3, ph + 1)
        plt.imshow(slice_np)
        plt.title(f'range: [{np.min(slice_np)}: {round(np.max(slice_np), 3)}]')
    plt.show()

def plot_ks_im(ks_real, ks_imag, recon):
    # recon = recon.numpy()
    plt.subplot(1, 3, 1)
    plt.imshow(ks_real, cmap='gray')
    plt.title('ks_real')
    plt.subplot(1, 3, 2)
    plt.imshow(ks_imag, cmap='gray')
    plt.title('ks_imag')
    plt.subplot(1, 3, 3)
    plt.imshow(recon, cmap='gray')
    plt.title('recon')
    plt.title(f'range: [{np.min(recon)}: {round(np.max(recon), 3)}]')
    plt.show()

# Normalize function (you can modify this based on the mode, e.g., 'zero_one', 'minus_one_one', etc.)
def normalize_data(data, mode='zero_one'):
    if mode == 'zero_one':
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    elif mode == 'minus_one_one':
        return 2 * (data - data.min()) / (data.max() - data.min() + 1e-8) - 1
    elif mode == 'z_score':
        return (data - data.mean()) / (data.std() + 1e-8)
    else:
        raise ValueError("Unsupported normalization mode. Choose 'zero_one', 'minus_one_one', or 'z_score'.")


def get_max_ps_length(ps_dir):
    max_length = 0
    for ps_file in os.listdir(ps_dir):
        if ps_file.endswith('.npy'):
            ps_data = np.load(os.path.join(ps_dir, ps_file))
            max_length = max(max_length, ps_data.shape[0])
    return max_length


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None  # Delay creation until forward pass
        self.device = None

    def gaussian_window(self, window_size, sigma):
        center = window_size // 2
        gauss = torch.tensor([
            math.exp(-(x - center) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, device, dtype):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device=device, dtype=dtype)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        # Rescale from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        _, channel, _, _ = pred.size()
        device = pred.device
        dtype = pred.dtype

        # Recreate window if necessary
        if self.window is None or self.channel != channel or self.device != device:
            self.window = self.create_window(self.window_size, channel, device, dtype)
            self.channel = channel
            self.device = device

        return 1 - self._ssim(pred, target, self.window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: weight for SSIMLoss. (1 - alpha) is weight for MSELoss.
        """
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIMLoss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target):
        ssim_er = self.ssim_loss(output, target)
        mse = self.mse_loss(output, target)
        return self.alpha * ssim_er + (1 - self.alpha) * mse




def save_generated_images(out_dir, output_img, target_img, epoch):
    """
    Save or plot the predicted (output_img) and ground truth (target_img) MR images during training.
    Supports both single image and batch input.
    Assumes images are normalized in [-1, 1].
    """
    os.makedirs(out_dir, exist_ok=True)

    def to_numpy(img_tensor):
        img_tensor = img_tensor.squeeze(0).cpu()  # [1, H, W] -> [H, W]
        img_tensor = img_tensor * 0.5 + 0.5  # Normalize from [-1, 1] to [0, 1]
        return img_tensor.detach().numpy()

    if output_img.dim() == 3:
        # Single image
        pred_img = to_numpy(output_img)
        gt_img = to_numpy(target_img)

        score = ssim(gt_img, pred_img, data_range=1.0)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'SSIM: {score:.4f}', fontsize=14)

        axs[0].imshow(gt_img, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')

        axs[1].imshow(pred_img, cmap='gray')
        axs[1].set_title('Prediction')
        axs[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"compare_epoch_{epoch + 1}.png"))
        plt.close()

    else:
        # Batch of images
        output_img = output_img.cpu() * 0.5 + 0.5
        target_img = target_img.cpu() * 0.5 + 0.5

        for idx in range(output_img.size(0)):
            pred_img = output_img[idx].squeeze(0).detach().numpy()
            gt_img = target_img[idx].squeeze(0).detach().numpy()

            score = ssim(gt_img, pred_img, data_range=1.0)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f'SSIM: {score:.4f}', fontsize=14)

            axs[0].imshow(gt_img, cmap='gray')
            axs[0].set_title('Ground Truth')
            axs[0].axis('off')

            axs[1].imshow(pred_img, cmap='gray')
            axs[1].set_title('Prediction')
            axs[1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(out_dir, f"compare_epoch_{epoch + 1}_img_{idx + 1}.png"))
            plt.close()

def save_generated_images_from_ks(out_dir, pred_kspace, target_kspace, epoch):
    """
    Converts predicted and target k-space [B, 2, H, W] to image domain, then saves and compares them.

    Args:
        pred_kspace (Tensor): [B, 2, H, W] - predicted k-space (real, imag)
        target_kspace (Tensor): [B, 2, H, W] - ground truth k-space (real, imag)
        epoch (int): Current epoch for filename
    """
    # os.makedirs(out_dir, exist_ok=True)




    # # Convert to complex k-space
    pred_complex = (pred_kspace[:, 0] + 1) / 2 * torch.exp (1j* torch.pi*pred_kspace[:, 1])  # [B, H, W]
    target_complex = (target_kspace[:, 0] + 1) / 2 * torch.exp (1j* torch.pi*target_kspace[:, 1])  # [B, H, W]

    # Inverse FFT to get image space
    pred_img = torch.fft.fftshift (torch.fft.ifft2(pred_complex))  # [B, H, W]
    target_img = torch.fft.fftshift (torch.fft.ifft2(target_complex)) # [B, H, W]

    # pred_np = np.abs(pred_img[0].cpu().numpy()) * 255
    # target_np = np.abs(target_img[0].cpu().numpy()) * 255
    # plt.subplot(1, 2, 1)
    # plt.imshow(target_np, 'gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(pred_np, 'gray')
    # score = ssim(target_np, pred_np, data_range=1.0)
    # plt.suptitle(f'SSIM: {score:.4f}', fontsize=14)
    # plt.show()

    # Plot each image in the batch
    B = pred_img.shape[0]
    for idx in range(B):
        pred_np = np.abs(pred_img[idx].cpu().numpy())
        target_np = np.abs(target_img[idx].cpu().numpy())
        score = ssim(target_np, pred_np, data_range=1.0)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'SSIM: {score:.4f}', fontsize=14)

        axs[0].imshow((target_np), cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')

        axs[1].imshow((np.log(pred_np)), cmap='gray')
        axs[1].set_title('Prediction')
        axs[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"compare_epoch_{epoch + 1}_img_{idx + 1}.png"))
        print('saved in', os.path.join(out_dir, f"compare_epoch_{epoch + 1}_img_{idx + 1}.png"))
        # plt.show()
        plt.close()


class CombinedKSpaceRowwiseMSELoss(nn.Module):
    def __init__(self, central_fraction=0.25,
                 real_weight=1.0, imag_weight=1.0,
                 central_weight=1.0, periphery_weight=1.0):
        """
        :param central_fraction: Fraction of k-space width to consider as central.
        :param real_weight: Weight for real part loss.
        :param imag_weight: Weight for imaginary part loss.
        :param central_weight: Weight for central region loss.
        :param periphery_weight: Weight for peripheral region loss.
        """
        super(CombinedKSpaceRowwiseMSELoss, self).__init__()
        self.central_fraction = central_fraction
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.central_weight = central_weight
        self.periphery_weight = periphery_weight
        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss()

    def compute_rowwise_mse(self, pred, target, B, H, W):
        central_width = int(W * self.central_fraction)
        start = (W - central_width) // 2
        end = start + central_width

        central_loss = 0.0
        periphery_loss = 0.0

        for b in range(B):
            for row in range(H):
                pred_row = pred[b, row, :]
                target_row = target[b, row, :]

                central_pred = pred_row[start:end]
                central_target = target_row[start:end]
                periphery_pred = torch.cat([pred_row[:start], pred_row[end:]])
                periphery_target = torch.cat([target_row[:start], target_row[end:]])

                central_loss += self.mse(central_pred, central_target)
                periphery_loss += self.mae(periphery_pred, periphery_target)

        total_rows = B * H
        central_loss /= total_rows
        periphery_loss /= total_rows

        return central_loss, periphery_loss

    def forward(self, pred, target):
        """
        :param pred: [B, 2, H, W] predicted k-space (real & imag)
        :param target: [B, 2, H, W] ground truth k-space
        :return: scalar combined loss
        """
        pred_real = pred[:, 0, :, :]
        target_real = target[:, 0, :, :]
        pred_imag = pred[:, 1, :, :]
        target_imag = target[:, 1, :, :]

        B, H, W = pred_real.shape

        real_central, real_periphery = self.compute_rowwise_mse(pred_real, target_real, B, H, W)
        imag_central, imag_periphery = self.compute_rowwise_mse(pred_imag, target_imag, B, H, W)

        real_loss = self.central_weight * real_central + self.periphery_weight * real_periphery
        imag_loss = self.central_weight * imag_central + self.periphery_weight * imag_periphery

        total_loss = self.real_weight * real_loss + self.imag_weight * imag_loss
        return total_loss

def compare_values_from_filename_and_json(filename, json_pth):
    """
    Compares values parsed from a filename with values extracted from a JSON file.

    Args:
        filename (str): The filename from which values will be extracted.
        json_file_path (str): The path to the JSON file from which values will be loaded.

    Returns:
        dict: A dictionary with the comparison results for each value.
    """
    # Parse values from the filename
    contrst =  (filename.split('_')[5])
    FoV_f = int (filename.split('_')[6] )/ 1000
    TR = int(filename.split('_')[7]) / 1000  # Convert TR from ms to seconds
    if contrst == 'T1w':
        N_TE = int(float (filename.split('_')[8]))
    else:
        N_TE = int(filename.split('_')[8])
    BW_pixel =  int((filename.split('_')[9].split('.json')[0]))

    # Load values from the JSON file
    with open(json_pth, 'r') as f:
        json_data = json.load(f)

    # Extract values from JSON
    json_FoV_f = json_data.get('FoV_f')
    json_TR = json_data.get('TR')
    json_N_TE = json_data.get('N_TE')
    json_BW_pixel = json_data.get('BW_pixel')

    # Compare and return results
    comparison_results = {
        'FoV_f': FoV_f == json_FoV_f,
        'TR': TR == json_TR,
        'N_TE': N_TE == json_N_TE,
        'BW_pixel': BW_pixel == json_BW_pixel,
    }
    # Check if all comparisons are True
    all_values_match = all(comparison_results.values())

    # Print the results
    if all_values_match:
        print("All values match!")
    else:
        print("Some values do not match!")
    return comparison_results
