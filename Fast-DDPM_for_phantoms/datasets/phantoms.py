import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

from .sr_util import get_paths_from_npys, brats_transform_augment, masks_transform


class Phantoms(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split

        gt_root = dataroot + '/masks/'
        img_root = dataroot + '/weighted/'
        self.img_npy_path, self.gt_npy_path = get_paths_from_npys(img_root, gt_root)
        self.data_len = len(self.img_npy_path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_FD = None
        img_LD = None
        base_name = None
        extension = None
        number = None
        FW_path = None
        BW_path = None

        base_name = self.img_npy_path[index].split('/')[-1]
        case_name = base_name.split('.')[0]
        case_name = case_name.split('\\')[-1]

        img_npy = np.load(self.img_npy_path[index])
        gt_npy = np.load(self.gt_npy_path[index])

        img, scale_img, offset_img = masks_transform(img_npy)

        gt, scale_gt, offset_gt = masks_transform(gt_npy)

        # Average values for the training dataset
        scale_gt = [0.0005511092837250057, 0.004553527884813633, 1.5115278426884828]
        offset_gt = [-1.0, -1.0, -1.0]

        return {'FD': gt, 'LD': img, 'case_name': case_name, 'scale_gt': scale_gt, 'offset_gt': offset_gt}


