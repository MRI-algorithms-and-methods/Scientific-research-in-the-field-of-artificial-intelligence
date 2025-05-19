# Fast-DDPM for MRI brain phantoms
 
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



