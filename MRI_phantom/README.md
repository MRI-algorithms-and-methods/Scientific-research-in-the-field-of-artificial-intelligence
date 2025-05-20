# MRI_phantom
<p align="center">
  <img src="https://github.com/user-attachments/assets/2afdd2ac-2a6e-4596-b5a0-879b6220d66b" alt="image" />
</p>

A digital MRI phantom represents a set of various tissue characteristics, such as T1 and T2 relaxation times and proton density. Digital phantoms simulate the properties and behavior of different tissues during an MRI procedure. They are an essential tool for developers and researchers in the field of MRI, allowing them to test and optimize imaging techniques, pulse sequences, and reconstruction algorithms without the need for physical phantoms, volunteer scanning, or actual MRI procedures.


This repository contains code for generating quantitative maps of various modalities (T1, T2, T2*, proton density) and assembling them into phantoms in HDF5 format, and code for reconstructing raw MRI data and obtaining a weighted image.
Here’s a concise version of your text:



MRI brain digital phantoms were generated using anatomical brain masks from the [BrainWeb database](https://brainweb.bic.mni.mcgill.ca/brainweb/). Each slice was segmented into 11 tissue types, including:

- Gray matter  
- White matter  
- Cerebrospinal fluid (CSF)  
- Fat  
- Muscle  
- Bone marrow  
- Skin  
- Dura mater  
- Vessels

Each tissue mask was assigned characteristic MRI properties—proton density (PD), T1, and T2 relaxation times—based on values reported in the literature. This resulted in realistic, ground truth digital phantoms suitable for simulating MRI acquisitions.

## Use Case: Creating MRI Digital Phantoms from BrainWeb Masks

This script (`simple_phantom_mask.py`) generates digital MRI brain phantoms based on anatomical segmentation masks from the [BrainWeb database](https://brainweb.bic.mni.mcgill.ca/brainweb/). It simulates tissue-specific quantitative MRI maps (e.g., T1, T2) by assigning physical parameters to each brain tissue.

### Key Features

- **Input:** Anatomical segmentation files (`.mnc` format) with 11 labeled tissues.
- **Tissues simulated:**  
  - CSF  
  - Gray Matter  
  - White Matter  
  - Fat  
  - Muscle  
  - Muscle/Skin  
  - Skull  
  - Vessels  
  - Around Fat  
  - Dura Matter  
  - Bone Marrow

- **Output:** Synthetic MRI slices saved as `.npy` files (NumPy arrays).
- **Customizable Parameters:** You can define tissue-specific T1, T2, or Proton Density (PD) distributions using:
  - `param_dict_*`: mean and standard deviation for each tissue.
  - `bound_dict_*`: value ranges for each tissue (used in boundary-based filling).
- **Resizing:** Each slice is resized to 128×128 pixels for consistency.
- **Phantom Creation Modes:**
  - `'homogeneous'` – uniform value per tissue
  - `'boundary_homogenous'` – random uniform value within bounds
  - `'boundary_heterogenous'` – voxel-wise random values within bounds
  - `'from_distribution'` – Gaussian-distributed values per tissue (default)

### Example Workflow

1. Define the quantitative parameter dictionary (e.g., `param_dict_T1`).
2. Set subject numbers for training and testing.
3. Run the `create_train_test()` function:
   - Loads `.mnc` anatomical models.
   - Extracts 2D slices with multiple tissue labels.
   - Calls `create_phantom()` for each valid slice.
   - Saves the result to `train_phantoms/` or `test_phantoms/` directories.

### Dependencies

- `numpy`
- `matplotlib`
- `nibabel`
- `PIL`
- `scikit-learn`
- `h5py` (optional, unused in this script)

### Sample Phantom Generation Function

```python
phantom = create_phantom(
    tissue_dict,
    param_dict_T1,
    have_params,
    data_slice,
    bound_dict=bound_dict_T1,
    proton_density=False,
    print_process=True,
    filling_type='from_distribution'
)


