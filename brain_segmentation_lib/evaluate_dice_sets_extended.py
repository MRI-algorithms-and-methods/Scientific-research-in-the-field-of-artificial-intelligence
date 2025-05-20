import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import nibabel as nib
from skimage.transform import resize
from brain_segmentation_lib.datasets.datasets import PatientDataLoader
from brain_segmentation_lib.utils.losses import multiclass_dice_loss
from brain_segmentation_lib.utils.metrics import multiclass_dice_coef

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

CLASS_NAMES = {1: "CSF", 2: "GM", 3: "WM"}




def multiclass_dice_coef_np(y_true, y_pred, smooth=1e-5):
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true + y_pred, axis=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return np.mean(dice)


def print_dice_scores(dices, tag=""):
    for i, d in enumerate(dices, start=1):
        print(f"{tag} Dice - {CLASS_NAMES[i]}: {round(d, 4)}")
    print(f"{tag} Mean Dice: {round(np.mean(dices), 4)}")


def evaluate_on_synthetic(patient_id, model_path):
    images, masks = PatientDataLoader.load_multiclass_patient_from_t1(
        t1_dir='F:/model/ml/testing/test_weighted_t1',
        masks_dir='F:/model/ml/testing/test_anatomic_masks',
        patient_id=patient_id
    )
    if images.shape[0] == 0:
        print(f"[Ошибка] Пациент {patient_id}: не найдено изображений для предсказания.")
        return
    y_true = to_categorical(masks.squeeze(), num_classes=4)
    model = load_model(model_path, custom_objects={
        'multiclass_dice_loss': multiclass_dice_loss,
        'multiclass_dice_coef': multiclass_dice_coef
    })
    y_pred = model.predict(images)
    dices = [multiclass_dice_coef_np(y_true[..., i], y_pred[..., i]) for i in range(1, 4)]
    print_dice_scores(dices, tag=f"Synthetic (patient {patient_id})")


def load_nifti(path):
    return nib.load(path).get_fdata()


def remap_mask_to_classes(mask, mapping):
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for original_label, new_label in mapping.items():
        new_mask[mask == original_label] = new_label
    return new_mask


def preprocess_multiclass(volume, mask, target_size=(128, 128), mapping=None):
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


def evaluate_on_real(patient_id, model_path):
    T1_PATH = os.path.join('F:/model/ml/training', 'training', f'{patient_id}', 'pre', 'reg_T1.nii.gz')
    MASK_PATH = os.path.join('F:/model/ml/training', 'training', f'{patient_id}', 'segm.nii.gz')
    if not os.path.exists(T1_PATH) or not os.path.exists(MASK_PATH):
        print(f"[Ошибка] Пациент {patient_id}: не найдено NIfTI.")
        return
    mapping = {5: 1, 6: 1, 1: 2, 2: 2, 3: 3, 4: 3}
    volume = load_nifti(T1_PATH)
    mask = load_nifti(MASK_PATH)
    X, y_true = preprocess_multiclass(volume, mask, mapping=mapping)
    model = load_model(model_path, custom_objects={
        'multiclass_dice_loss': multiclass_dice_loss,
        'multiclass_dice_coef': multiclass_dice_coef
    })
    y_pred = model.predict(X)
    dices = [multiclass_dice_coef_np(y_true[..., i], y_pred[..., i]) for i in range(1, 4)]
    print_dice_scores(dices, tag=f"Real (patient {patient_id})")
import numpy as np
import cv2
from typing import Tuple, Union

def scale_keep_res(
        image: np.ndarray,
        mask:  np.ndarray,
        scale: Union[float, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Равномерно или анизотропно (по ширине / высоте) масштабирует
    изображение и маску, возвращая результат в исходном разрешении.

    Parameters
    ----------
    image : np.ndarray
        Оригинальное изображение (H, W[, C]).
    mask  : np.ndarray
        Соответствующая маска (H, W) или (H, W, 1).
    scale : float | (float, float)
        • float  ― единый коэффициент для обеих осей,
        • tuple ― (scale_y, scale_x) для независимого масштабирования.

        <1.0 → сжатие, >1.0 → растяжение.

    Returns
    -------
    img_out, mask_out : Tuple[np.ndarray, np.ndarray]
        Масштабированные изображение и маска исходного размера.
    """
    # --- 0. Нормализуем аргумент scale ---
    if isinstance(scale, (int, float)):
        scale_y = scale_x = float(scale)
    elif isinstance(scale, (tuple, list)) and len(scale) == 2:
        scale_y, scale_x = map(float, scale)
    else:
        raise ValueError("scale должен быть float или (float, float)")

    h, w = image.shape[:2]
    new_h, new_w = max(1, int(round(h * scale_y))), max(1, int(round(w * scale_x)))

    # --- 1. Ресайз ---
    interp_img  = cv2.INTER_LINEAR  if image.dtype != np.uint8 else cv2.INTER_AREA
    interp_mask = cv2.INTER_NEAREST
    img_scaled  = cv2.resize(image, (new_w, new_h), interpolation=interp_img)
    mask_scaled = cv2.resize(mask,  (new_w, new_h), interpolation=interp_mask)

    # --- 2. Падд / кроп до исходного размера ---
    # 2a. Если новая картинка меньше по высоте/ширине → паддинг
    pad_top = (h - new_h) // 2 if new_h <= h else 0
    pad_left = (w - new_w) // 2 if new_w <= w else 0
    pad_bottom = h - new_h - pad_top if new_h <= h else 0
    pad_right  = w - new_w - pad_left if new_w <= w else 0

    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        img_scaled = cv2.copyMakeBorder(img_scaled, pad_top, pad_bottom,
                                        pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=0)
        mask_scaled = cv2.copyMakeBorder(mask_scaled, pad_top, pad_bottom,
                                         pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=0)

    # 2b. Если изображение стало больше → центральный кроп
    start_y = (img_scaled.shape[0] - h) // 2
    start_x = (img_scaled.shape[1] - w) // 2
    img_out  = img_scaled[start_y:start_y + h, start_x:start_x + w]
    mask_out = mask_scaled[start_y:start_y + h, start_x:start_x + w]

    # --- 3. Контроль качества ---
    assert img_out.shape[:2] == (h, w) and mask_out.shape[:2] == (h, w), \
        "Размеры не совпали с исходными"

    return img_out, mask_out



def evaluate_on_mixed(synth_patient_id, real_patient_id, model_path):
    X_synth, m_synth = PatientDataLoader.load_multiclass_patient_from_t1(
        'F:/model/ml/testing/test_weighted_t1',
        'F:/model/ml/testing/test_anatomic_masks',
        synth_patient_id
    )
    y_synth = to_categorical(m_synth.squeeze(), num_classes=4)

    T1_PATH = os.path.join('F:/model/ml/training', 'training', f'{real_patient_id}', 'pre', 'reg_T1.nii.gz')
    MASK_PATH = os.path.join('F:/model/ml/training', 'training', f'{real_patient_id}', 'segm.nii.gz')
    mapping = {5: 1, 6: 1, 1: 2, 2: 2, 3: 3, 4: 3}
    vol = load_nifti(T1_PATH)
    msk = load_nifti(MASK_PATH)
    X_real, y_real = preprocess_multiclass(vol, msk, mapping=mapping)

    # Пример изображения и маски
    image = X_real[0].squeeze()  # X — массив изображений (128x128x1)
    mask = np.argmax(y_real[0], axis=-1)  # y — one-hot маски (128x128x4)

    augmentations = [
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
    ]

    # Визуализация
    plt.figure(figsize=(6, 4))
    for i, aug in enumerate(augmentations):
        augmented = aug(image=image, mask=mask)
        img_aug = augmented['image']
        msk_aug = augmented['mask']

        img_aug_new,msk_aug_new = scale_keep_res(img_aug, msk_aug, 0.7)
        # msk_aug_new = scale_keep_res(msk_aug, 1, 0.7)

        # img_aug_new = img_aug
        # msk_aug_new = msk_aug
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_aug_new, cmap='gray')
        plt.title(f'Image {i}')
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        plt.imshow(msk_aug_new, cmap='gray')
        plt.title(f'Mask {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("X_synth shape:", X_synth.shape)
    print("X_real shape:", X_real.shape)
    X = np.concatenate([X_synth, X_real])
    y = np.concatenate([y_synth, y_real])

    # model = load_model(model_path, custom_objects={
    #     'multiclass_dice_loss': multiclass_dice_loss,
    #     'multiclass_dice_coef': multiclass_dice_coef
    # })
    # y_pred = model.predict(X)
    # dices = [multiclass_dice_coef_np(y[..., i], y_pred[..., i]) for i in range(1, 4)]
    # print_dice_scores(dices, tag=f"Mixed (synth {synth_patient_id} + real {real_patient_id})")





MODEL_PATH = "unet_trained_on_mixed_v1.h5"
# evaluate_on_synthetic(18, MODEL_PATH)
# evaluate_on_real(29, MODEL_PATH)
evaluate_on_mixed(18, 29, MODEL_PATH)


