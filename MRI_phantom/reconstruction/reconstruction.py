import numpy as np
import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from PIL import Image
import os

# Reconstriction of raw MRI data (in HDF5 format)

# Example of k_space_order
#k_space_order = [64, 0, 65, 1, 66, 2, 67, 3, 68, 4, 69, 5, 70, 6,
                #71, 7, 72, 8, 73, 9, 74, 10, 75, 11, 76, 12, 77, 13,
                #78, 14, 79, 15, 80, 16, 81, 17, 82, 18, 83, 19, 84, 20,
                #85, 21, 86, 22, 87, 23, 88, 24, 89, 25, 90, 26, 91, 27,
                #92, 28, 93, 29, 94, 30, 95, 31,
                #32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102,
                #39, 103, 40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109,
                #46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51, 115, 52, 116,
                #53, 117, 54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123,
                #60, 124, 61, 125, 62, 126, 63, 127]

def process_hdf5(input_filename, output_filename, k_space_order):
    with h5py.File(input_filename, 'r') as f:
        key = list(f.keys())[0]
        data = f[key]
        keyw = list(data.keys())[0]
        arr_data = np.array(data[keyw])


    sorted_arr_data = np.zeros_like(arr_data)
    for i, source_idx in enumerate(k_space_order):
        sorted_arr_data[:,source_idx, :] = arr_data[:, i, :]

    reconstructed_data = []

    for slice_idx in range(sorted_arr_data.shape[2]):
        matrix = sorted_arr_data[:, :, slice_idx]
        fft_shifted_data = np.fft.fftshift(np.fft.fft2(matrix))
        amplitude = np.abs(fft_shifted_data)
        reconstructed_data.append(np.copy(fft_shifted_data))

        plt.imshow(amplitude, cmap='gray', vmin=amplitude.min(), vmax=amplitude.max())
        plt.colorbar()
        plt.show()

    np.save(output_filename, np.array(reconstructed_data))
    print(f"File save: {output_filename}")

