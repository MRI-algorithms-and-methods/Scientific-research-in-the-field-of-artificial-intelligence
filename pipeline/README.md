# 🧠 Pipeline Overview: Synthetic MRI Dataset Generation (Non AI)

![image](https://github.com/user-attachments/assets/c2888572-c93d-4d52-bd8e-c6e5373bd177)



## 1. Digital Phantom Creation

Utilize anatomically realistic digital phantoms which offer high-resolution representations of human anatomy with detailed MRI properties.  We created MRI brain pantoms with tissue properties (T1, T2, T2*, proton density) based on humain brain tissues dataest from **BrainWeb** [Source](https://brainweb.bic.mni.mcgill.ca/brainweb/)
 

## 2. Pulse Sequence Design

Design MRI pulse sequences using the **Pulseq** framework, allowing for hardware-independent and vendor-neutral sequence prototyping. [Source](https://pulseq.github.io/)


## 3. MRI Simulation with KomaMRI.jl

Employ **KomaMRI.jl**, an open-source MRI simulation framework developed in Julia, compatible with Pulseq sequences. KomaMRI.jl efficiently solves the Bloch equations using CPU and GPU parallelization, enabling realistic simulation of MRI acquisitions. [Source](https://pubmed.ncbi.nlm.nih.gov/36877139/)
We implement the simulation environment within a Docker container, ensuring reproducibility and ease of deployment across different systems. This encapsulation allows for consistent simulation setups and facilitates integration into various workflows.

## 4. K-Space Data Generation and Sorting

Simulate raw MRI data acquisition, resulting in raw k-space data  in the ISMRMRD format. Subsequently, sort the k-space data to prepare it for image reconstruction processes. We can save the sorted K-space for their furthur use in tasks related to k-space domain (such as MR image reconstruction)

## 5. Image Reconstruction

Reconstruct MR images from the sorted k-space data using appropriate reconstruction algorithms, such as those provided by **MRIReco.jl**, ensuring high-fidelity image outputs. [Source](https://pubmed.ncbi.nlm.nih.gov/33817833/)

# 📘 Use Case: Simulating MRI with `simulate_mri.py`

This script allows you to simulate **T1**, **T2**, or **PD**-weighted MRI images from phantom data and pulse sequences.

---

## 🧩 Inputs

To run the simulation, you need to provide:

- **Phantom Data Folder** (`*.h5`):  
  Contains digital anatomical phantoms.

- **Pulse Sequences Folder** (`*.seq`):  
  Contains MRI pulse sequences defining acquisition strategies.

- **K-space Order File** (`*.npy` or other):  
  Describes the order in which k-space is filled. Required for sorting.

---

## 📤 Outputs

The script will generate:

- **Raw K-space data** (unsorted)
- **Sorted K-space data** (based on the order file)
- **Reconstructed image files** (`.png`, `.jpg`, etc.)
- **Reconstructed arrays** (`.npy` format)

---

## 🗂️ Example Input and Output Paths

```python
base_folder = r'E:\Dataset'
contrast = 'PD'  # Choose from 'T1', 'T2', or 'PD'
split_type = 'train'  # e.g., 'train', 'val', 'test'

# Input paths
phantoms_pth = fr'{base_folder}\{contrast}\phantoms'
seq_dir = fr'{base_folder}\{contrast}\PS\seq'
kso_pth = fr'{base_folder}\{contrast}\PS\ks_order'

# Output paths
output_ks_raw = fr'{base_folder}\{split_type}\{contrast}\kspace\raw'
output_ks_sorted = fr'{base_folder}\{split_type}\{contrast}\kspace\sorted'
output_im = fr'{base_folder}\{split_type}\{contrast}\recon\imges'
output_rec_npy = fr'{base_folder}\{split_type}\{contrast}\recon\npy'
```
## You can run the simulation script with the following arguments:
```
python simulate_mri.py \
  --phantoms_pth "E:\Dataset\PD\phantoms" \
  --seq_dir "E:\Dataset\PD\PS\seq" \
  --kso_pth "E:\Dataset\PD\PS\ks_order" \
  --output_ks_raw "E:\Dataset\train\PD\kspace\raw" \
  --output_ks_sorted "E:\Dataset\train\PD\kspace\sorted" \
  --output_img "E:\Dataset\train\PD\recon\imges" \
  --output_npy "E:\Dataset\train\PD\recon\npy" \
  --contrast "PD"


