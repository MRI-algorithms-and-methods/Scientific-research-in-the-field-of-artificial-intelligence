import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json

def order_ks_extract(ks_order_file):
    # Open and read the JSON file containing the k-space order
    with open(ks_order_file, "r") as file:
        data = json.load(file)  # Parse JSON data into a Python dictionary

    # Extract the nested list of k-space indices
    order = data["k_space_order"]

    # Flatten the nested list (e.g., [[1, 2], [3, 4]] -> [1, 2, 3, 4])
    flat_order = [item for sublist in order for item in sublist]

    # Convert all valid entries to integers
    # This removes non-numeric strings and keeps only valid digit strings or ints
    final_order = [
        int(x) for x in flat_order 
        if (isinstance(x, str) and x.isdigit()) or isinstance(x, int)
    ]

    # Return the flattened and cleaned list of k-space indices
    return final_order


def process_hdf5_kspace(input_filename, ks_order_file=None, show_image=False):
    """
    Process HDF5 k-space data with or without sorting. Then reconstruct MRI
    
    Parameters:
        input_filename (str): Path to HDF5 file.
        ks_order_file (str, optional): Path to JSON file specifying k-space order.
        show_image (bool): If True, display reconstructed image.
    
    Returns:
        tuple:
            - sorted or original k-space (2D numpy array)
            - reconstructed image (2D numpy array)
    """
    # Load the array from the HDF5 file
    with h5py.File(input_filename, 'r') as f:
        group = next(iter(f.values()))
        dataset = next(iter(group.values()))
        arr_data = np.array(dataset)

    if ks_order_file:
        # Reorder k-space lines using provided order
        k_space_order = order_ks_extract(ks_order_file)
        sorted_ks = np.zeros_like(arr_data)
        for i, target_idx in enumerate(k_space_order):
            sorted_ks[:, target_idx, :] = arr_data[:, i, :]
        kspace_data = np.squeeze(sorted_ks)
    else:
        # Use first channel without reordering
        kspace_data = arr_data[:, :, 0]

    # Reconstruct image using 2D FFT
    fft_shifted = np.fft.fftshift(np.fft.fft2(kspace_data))
    amplitude = np.abs(fft_shifted)
    reconstructed_img = amplitude.T

    if show_image:
        plt.imshow(reconstructed_img, cmap='gray', vmin=amplitude.min(), vmax=amplitude.max())
        plt.colorbar()
        plt.title("Reconstructed FFT Magnitude")
        plt.show()

    return kspace_data, reconstructed_img
