# Fast-DDPM for MRI brain phantoms

This repository contains the official implementation of the following paper: [Deep Learning-Based creating of a Digital MRI Phantom for the Brain](https://doi.org/10.1007/s10334-025-01278-8)
 
A model for generating quantitative MRI maps of The Brain

[This implementation is based on / inspired by: Fast-DDPM](https://arxiv.org/abs/2405.14802)

The code is only for research purposes. 

## Requirements
* Python==3.10.6
* torch==1.12.1
* torchvision==0.15.2
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* scikit-image
* medpy
* pillow
* scipy
* `pip install -r requirements.txt`


## Usage
### 1. Git clone or download the codes.

### 2. Pretrained model weights
* We provide pretrained model weights, you can write to Kseniya Belousova (kseniia.belousova@metalab.ifmo.ru)


### 3. Prepare data
There is one sample for testing the model (two sets of T1, T2, PD quantitative maps and two sets of T1w, T2w, PDw weighted images)
data\test\masks
data\test\weighted



### 4. Training/Sampling a Fast-DDPM model

```
python fast_ddpm_main.py --config phantoms_linear.yml --dataset PHANTOMS --timesteps 1000
```
```
python fast_ddpm_main.py --config phantoms_linear.yml --dataset PHANTOMS --sample --fid --timesteps 1000
```


## Citation
If you use this this repository, please cite the following paper:


> **Kseniya Belousova, Zilya Badrieva, Iuliia Pisareva, Nikita Babich, Dmitriy Agapov, Olga Pavlova, Ekaterina Brui, Walid Al-Haidri**
> 
> *Deep learning-based generation of a digital MRI brain phantom*
> 
> Book of Abstracts ESMRMB 2025 Online 41st Annual Scientific Meeting 8–11 October 2025. Magn Reson Mater Phy (2025).
> 
> https://doi.org/10.1007/s10334-025-01278-8
