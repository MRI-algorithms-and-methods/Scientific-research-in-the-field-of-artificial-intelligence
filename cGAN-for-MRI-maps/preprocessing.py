import os
import numpy as np

weighted_dir = "/data/train/weighted"
map_dir = "/data/train/masks"

def normalize_channels(img):
    img = img.astype(np.float32)
    img_norm = np.zeros_like(img)
    params = []

    for c in range(img.shape[-1]):
        ch = img[..., c]
        ch_min = ch.min()
        ch_max = ch.max()
        scale = 2.0 / (ch_max - ch_min) if ch_max > ch_min else 1.0
        offset = -1.0 - ch_min * scale
        img_norm[..., c] = ch * scale + offset
        params.append((ch_min, ch_max, scale, offset))

    return img_norm, params

def load_real_samples(train_weighted_dir, train_map_dir):
    weighted_files = set(f for f in os.listdir(train_weighted_dir) if f.endswith('.npy'))
    map_files = set(f for f in os.listdir(train_map_dir) if f.endswith('.npy'))
    common_files = list(weighted_files & map_files)
    common_files.sort()

    X1, X2 = [], []
    params_weighted = {}
    params_map = {}

    for filename in common_files:
        weighted = np.load(os.path.join(train_weighted_dir, filename))
        t_map = np.load(os.path.join(train_map_dir, filename))

        # Rotate
        # weighted = np.rot90(weighted, k=3)
        # t_map = np.rot90(t_map, k=3)

        weighted_norm, w_params = normalize_channels(weighted)
        t_map_norm, t_params = normalize_channels(t_map)

        X1.append(weighted_norm)
        X2.append(t_map_norm)

        params_weighted[filename] = w_params
        params_map[filename] = t_params

    X1 = np.array(X1)
    X2 = np.array(X2)

    return [X1, X2], (params_weighted, params_map)

dataset, scale_param = load_real_samples(weighted_dir, map_dir)

X1, X2 = dataset
params_weighted, params_map = scale_param

save_path = '/data/mri_dataset_all.npz'

np.savez_compressed(
    save_path,
    X1=X1,
    X2=X2,
    params_weighted=np.array(params_weighted, dtype=object),
    params_map=np.array(params_map, dtype=object)
)
