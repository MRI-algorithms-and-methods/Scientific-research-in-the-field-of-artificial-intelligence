# 🧠 Synthetic MRI Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-Available-brightgreen.svg)](https://drive.google.com/file/d/1cIJ-pzP3Sd16GqXRqOqlJ0tyiZ0slTiW/view?usp=sharing)
[![Made with ❤️](https://img.shields.io/badge/ITMO%20-%E2%9D%A4-red)](https://itmo.ru/)

## 📌 Overview

This repository contains a **synthetic MRI dataset** with **23,329 samples** across **T1, T2, and PD contrasts**, split into:

- **Train**: 16,867 samples  
- **Test**: 6,462 samples

> ⚠️ **Note**: Due to GitHub file size restrictions, only a few **example files** are included.  
> The full dataset is available via the link below.

🔗 **[Download Full Dataset](https://drive.google.com/file/d/1cIJ-pzP3Sd16GqXRqOqlJ0tyiZ0slTiW/view?usp=sharing)** <!-- Replace # with your actual link -->


## 📂 Dataset Structure

The dataset is organized into `train/` and `test/` folders with identical subfolder structures:

### 1. `k_space/`
Contains raw and sorted k-space data.

- `raw/`:  
  `.h5` files with **raw k-space data**

- `sorted/`:  
  `.npy` files with **k-space sorted by the filling order**  
  (see `ps/ks_order/*.json`)



### 2. `phantoms/`
Phantom data in `.h5` format with shape `(250, 250, 5)`:

- Channels: **T1, T2, T2\*, PD, D**



### 3. `ps/` (Pulse Sequences)

- `parameters/`:  
  `.json` files containing pulse sequence parameters

- `seq/`:  
  `.seq` files representing pulse sequences

- `wave/`:  
  `.npy` files of interpreted sequences as waveforms

- `ks_order/`:  
  `.json` files defining the k-space filling order

- `param_encode/`:  
  Digitally encoded pulse sequence parameters for DL model input



### 4. `reconstructed/`

Reconstructed MR images in two formats:

- `.png`: For visualization  
- `.npy`: For DL training/testing



## 🚀 Example Usage

```python
import h5py
import numpy as np

# Load phantom
phantom = h5py.File("phantoms/sample.h5", "r")["phantom"][:]

# Load sorted k-space
kspace = np.load("k_space/sorted/sample.npy")

# Load pulse sequence waveform
waveform = np.load("ps/wave/sample.npy")
```

## Disclaimer

It is expressly understood that **ITMO University** does not make any warranties regarding the synthetic MRI data provided. Specifically, ITMO University does **not warrant or guarantee** that:

- The data is accurate;
- The data is merchantable;
- The data is fit for any particular purpose, including but not limited to clinical, diagnostic, or commercial applications.

**Use of this data is entirely at your own risk.**

ITMO University **shall not be held liable** for any claims, damages, or losses incurred by you or any third party as a result of using the data, whether directly or indirectly.

By accessing or using the synthetic MRI data, you acknowledge and accept these terms.

---

### License

This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. You are free to:

- **Share** — copy and redistribute the material in any medium or format;
- **Adapt** — remix, transform, and build upon the material;

**Under the following terms:**

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **Scientific Use Only** — Use of this dataset is restricted to non-commercial **scientific research and educational purposes**.



