Synthetic MRI Dataset
📌 Overview
This repository contains a synthetic MRI dataset with 23,329 samples (T1, T2, and PD contrasts), split into:

Train: 16,867 samples

Test: 6,462 samples

Due to GitHub file size restrictions, only example files are included here. The full dataset is available via the link below.

🔗 Download Full Dataset: [Insert Link Here]

📂 Dataset Structure
The dataset is organized into train and test folders with identical structures.

1. k_space
raw/:

.h5 files containing raw k-space data.

sorted/:

.npy files with k-space data sorted using the filling order (see ps/ks_order/*.json).

2. phantoms
.h5 files with phantom data (shape: (250, 250, 5)), including 5 channels:

T1, T2, T2*, PD, and D.

3. ps (Pulse Sequence)
parameters/:

.json files with pulse sequence parameters.

seq/:

.seq pulse sequence files.

wave/:

.npy files representing interpreted pulse sequences in waveform style.

ks_order/:

.json files defining k-space filling order.

param_encode/:

Digitally encoded pulse sequence parameters for DL model input.

4. reconstructed
Reconstructed MR images in two formats:

.png for visualization.

.npy for DL training/testing.