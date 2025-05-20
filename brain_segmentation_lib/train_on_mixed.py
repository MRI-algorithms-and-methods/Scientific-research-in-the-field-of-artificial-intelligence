import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from brain_segmentation_lib.datasets.datasets import PatientDataLoader
from brain_segmentation_lib.utils.losses import multiclass_dice_loss
from brain_segmentation_lib.utils.metrics import multiclass_dice_coef
from matplotlib import pyplot as plt
from brain_segmentation_lib.models.unet import build_multiclass_unet

def load_nifti(path):
    return nib.load(path).get_fdata()

def remap_mask_to_classes(mask, mapping):
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for original_label, new_label in mapping.items():
        new_mask[mask == original_label] = new_label
    return new_mask

def preprocess_multiclass(volume, mask, target_size=(256, 256), mapping=None):
    X = []
    y = []
    for i in range(volume.shape[2]):
        slice_img = np.rot90(volume[:, :, i], k=-1)
        slice_mask = np.rot90(mask[:, :, i], k=-1)
        if mapping:
            slice_mask = remap_mask_to_classes(slice_mask, mapping)
        img_resized = resize(slice_img, target_size, preserve_range=True, anti_aliasing=True)
        mask_resized = resize(slice_mask, target_size, preserve_range=True, anti_aliasing=False, order=0)
        img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))
        X.append(img_norm[..., np.newaxis])
        y.append(mask_resized.astype(np.uint8))
    return np.array(X), to_categorical(np.array(y), num_classes=4)

def random_subset(X, y, percent=0.3, seed=None):
    assert 0 < percent <= 1, "Процент должен быть в пределах (0, 1]"
    n = len(X)
    k = int(n * percent)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    return X[indices], y[indices]

def load_synthetic_patients(patient_ids):
    images_list, masks_list = [], []
    for pid in patient_ids:
        imgs, msks = PatientDataLoader.load_multiclass_patient_from_t1(
            t1_dir='F:/model/ml/training/train_weighted_t1',
            masks_dir='F:/model/ml/training/train_anatomic_masks',
            patient_id=pid
        )
        images_list.append(imgs)
        masks_list.append(msks)
    images = np.concatenate(images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    return images, to_categorical(masks.squeeze(), num_classes=4)

import albumentations as A

def get_augmentation():
    return A.Compose([
        A.RandomRotate90(p=1.0)
    ])

def load_real_patients(patient_ids, augment=False,num_augments=4):
    mapping = {5: 1, 6: 1, 1: 2, 2: 2, 3: 3, 4: 3}
    images_all, masks_all = [], []
    aug = get_augmentation() if augment else None

    for pid in patient_ids:
        t1_path = os.path.join('F:/model/ml/training/', 'training', f'{pid}', 'pre', 'reg_T1.nii.gz')
        mask_path = os.path.join('F:/model/ml/training/', 'training', f'{pid}', 'segm.nii.gz')

        if os.path.exists(t1_path) and os.path.exists(mask_path):
            volume = load_nifti(t1_path)
            mask = load_nifti(mask_path)
            X, y = preprocess_multiclass(volume, mask, mapping=mapping)

            if augment:
                X_aug, y_aug = [], []
                for img, msk in zip(X, np.argmax(y, axis=-1)):
                    for _ in range(num_augments):
                        augmented = aug(image=img.squeeze(), mask=msk)
                        X_aug.append(augmented['image'][..., np.newaxis])
                        y_aug.append(to_categorical(augmented['mask'], num_classes=4))
                X = np.array(X_aug)
                y = np.array(y_aug)

            images_all.append(X)
            masks_all.append(y)
        else:
            print(f"[Пропущен] Пациент {pid} — нет данных.")

    if not images_all or not masks_all:
        raise ValueError("Не удалось загрузить ни одного пациента из real_ids.")

    return np.concatenate(images_all), np.concatenate(masks_all)

def train_on_mixed(synth_ids, real_ids, synth_percent=0.5, model_out_path="unet_mixed_trained.h5"):
    X_synth, y_synth = load_synthetic_patients(synth_ids)
    X_real, y_real = load_real_patients(real_ids, augment=True, num_augments=5)

    X_synth_small, y_synth_small = random_subset(X_synth, y_synth, percent=synth_percent)

    X = np.concatenate([X_synth_small, X_real])
    y = np.concatenate([y_synth_small, y_real])

    print(f"Общий размер выборки: {X.shape[0]} сэмплов")

    model = build_multiclass_unet(input_shape=(128, 128, 1), num_classes=4)
    model.compile(optimizer=Adam(1e-4), loss=multiclass_dice_loss, metrics=[multiclass_dice_coef])

    history = model.fit(X, y, batch_size=8, epochs=70, validation_split=0.1)

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['multiclass_dice_coef'], label='Train Dice')
    plt.plot(history.history['val_multiclass_dice_coef'], label='Val Dice')
    plt.title("Коэффициент Дайса. Обучение на реальных данных")
    plt.xlabel("Эпоха")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid()
    plt.savefig("history_mixed_256.png")
    plt.show()

    model.save(model_out_path)
    print(f"Модель сохранена в {model_out_path}")

# Пример запуска
train_on_mixed(synth_ids=[4, 5, 6], real_ids=[1,4,5,7,14,27], synth_percent=0.6, model_out_path="unet_trained_on_mixed_256.h5")
