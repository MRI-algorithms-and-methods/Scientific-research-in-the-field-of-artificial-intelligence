import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from brain_segmentation_lib.datasets.datasets import PatientDataLoader
from brain_segmentation_lib.visualization.visualization import Visualizer


def visualize_class_mapping(predictions, true_masks, class_names=["CSF", "GM", "WM", "BG"]):
    """
    Визуализирует предсказания и реальные маски с отображением номеров классов.
    """
    plt.figure(figsize=(16, 4))

    for i, class_name in enumerate(class_names):
        # Извлекаем маску для этого класса
        pred_class = (predictions == i).astype(np.uint8)
        true_class = (true_masks == i).astype(np.uint8)

        # Субплот для каждого класса
        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(pred_class, cmap="gray")
        plt.title(f"Predicted {class_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_class_mapping_synthetic(predictions, true_masks, class_names=["CSF", "GM", "WM", "BG"]):
    """
    Визуализирует предсказания для синтетических данных.
    """
    plt.figure(figsize=(16, 4))

    for i, class_name in enumerate(class_names):
        # Извлекаем маску для этого класса
        pred_class = (predictions == i).astype(np.uint8)
        true_class = (true_masks == i).astype(np.uint8)

        # Субплот для каждого класса
        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(true_class, cmap="gray")
        plt.title(f"True Mask {class_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


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

        slice_img_resized = resize(slice_img, target_size, preserve_range=True, anti_aliasing=True)
        slice_mask_resized = resize(slice_mask, target_size, preserve_range=True, anti_aliasing=False, order=0)

        slice_img_norm = (slice_img_resized - np.min(slice_img_resized)) / (
                    np.max(slice_img_resized) - np.min(slice_img_resized))

        X.append(slice_img_norm[..., np.newaxis])
        y.append(slice_mask_resized.astype(np.uint8))

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=4)
    return X, y


def plot_prediction(image, pred_mask, class_names=["BG", "CSF", "GM", "WM"]):
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("T1 Image")
    plt.axis("off")

    for i in range(1, 4):
        plt.subplot(1, 4, i + 1)
        plt.imshow((pred_mask == i).astype(np.uint8), cmap="gray")
        plt.title(f"Predicted {class_names[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("finetuned_result.png")
    plt.show()


def multiclass_dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true + y_pred, axis=[0, 1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)


def multiclass_dice_loss(y_true, y_pred):
    return 1 - multiclass_dice_coef(y_true, y_pred)


# Конфигурация
patient_ids = [1, 4, 5, 7, 14]  # сюда добавь нужные номера пациентов

PRETRAINED_MODEL_PATH = 'unet_trained_on_nifti_v2.h5'
FINETUNED_MODEL_PATH = 'unet_finetuned_first_real_then_synth_v2.h5'
NEW_MODEL_PATH = 'unet_realdata_v2.h5'

# Маппинг классов
mapping = {
    5: 1, 6: 1,  # CSF
    1: 2, 2: 2,  # GM
    3: 3, 4: 3  # WM
}

# сначала реальные, потом синтетические
masks_dir = 'F:/model/ml/training/train_anatomic_masks'
t1_dir = 'F:/model/ml/training/train_weighted_t1'
test_masks_dir = 'F:/model/ml/testing/test_anatomic_masks'
testing_t1_dir = 'F:/model/ml/testing/test_weighted_t1'

# загрузка данных для пациентов 4, 5, 6
images_list, masks_list = [], []
for patient_id in [4, 5, 6]:
    imgs, msks = PatientDataLoader.load_multiclass_patient_from_t1(t1_dir, masks_dir, patient_id)
    images_list.append(imgs)
    masks_list.append(msks)

images = np.concatenate(images_list, axis=0)
masks = np.concatenate(masks_list, axis=0)

# plt.imshow(images[50])
masks_cat = to_categorical(masks.squeeze(), num_classes=4)

print("After augmentation:", images.shape, masks.shape)
print("Images shape:", images.shape)
print("Masks one-hot shape:", masks_cat.shape)

# # визуализация ground truth
# for i in range(50,52):
#     Visualizer.plot_multiclass_mask(images[i], masks[i], class_names=["BG", "CSF", "GM", "WM"])

X, y = images, masks_cat
# y = y[..., 1:]

# model = load_model(PRETRAINED_MODEL_PATH, custom_objects={
#     'multiclass_dice_loss': multiclass_dice_loss,
#     'multiclass_dice_coef': multiclass_dice_coef
# })

# сначала синтетические, потом реальные
real_data = []
real_masks = []

for pid in patient_ids:
    t1_path = os.path.join('F:/model/ml/training', 'training', f'{pid}', 'pre', 'reg_T1.nii.gz')
    mask_path = os.path.join('F:/model/ml/training', 'training', f'{pid}', 'segm.nii.gz')

    volume = load_nifti(t1_path)
    mask = load_nifti(mask_path)

    X_pid, y_pid = preprocess_multiclass(volume, mask, mapping=mapping)
    real_data.append(X_pid)
    real_masks.append(y_pid)

# Объединение всех пациентов в общий массив
real_data_ready = np.concatenate(real_data, axis=0)
real_masks_ready = np.concatenate(real_masks, axis=0)

print("shape real data" + str(np.shape(real_data_ready)))
print("shape real masks" + str(np.shape(real_masks_ready)))

# # Загрузка и компиляция модели
# model.compile(optimizer=Adam(1e-4), loss=multiclass_dice_loss, metrics=[multiclass_dice_coef])
#
# # Дообучение
# history = model.fit(X, y, batch_size=8, epochs=30, validation_split=0.1)
#
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.plot(history.history['multiclass_dice_coef'], label='Train Dice')
# plt.plot(history.history['val_multiclass_dice_coef'], label='Val Dice')
# plt.title("Дообучение на синтетических данных.")
# plt.xlabel("Эпоха")
# plt.ylabel("Dice")
# plt.legend()
# plt.grid()
# plt.savefig("history_tuned_realdata_50+30_ep.png")
# plt.show()

# # Сохранение дообученной модели
# model.save(FINETUNED_MODEL_PATH)



import cv2
import numpy as np
from typing import Tuple, Optional

def adjust_fov_scale(
    img: np.ndarray,
    current_fov: float,
    target_fov: float,
    *,
    center: Optional[Tuple[int, int]] = None,
    keep_resolution: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | Tuple[int, int, int] = 0,
) -> np.ndarray:
    """
    Униформ‑масштаб изображения → новый FoV (горизонтальный = вертикальный).

    current_fov и target_fov — диагональные, горизонтальные или вертикальные —
    неважно: важна только их <отношение>.

    🔎 Логика
    --------
    FoV ∝ 1 / zoom.
        - target < current  →  zoom‑IN  (кроп → ресайз)
        - target > current  →  zoom‑OUT (сжатие → паддинг)

    Параметры
    ---------
    img : np.ndarray
        BGR или Gray кадр.
    current_fov, target_fov : float
        Текущий и желаемый углы обзора (°).
    center : (x, y) | None
        Точка «фокуса». По умолчанию — геометрический центр.
    keep_resolution : bool
        True  → итоговое изображение такого же H×W,
        False → отдаём результат после кропа/скейла как есть.
    interpolation : int
        Алгоритм в cv2.resize.
    border_mode, border_value
        Чем заполнять рамки при zoom‑OUT.

    Возврат
    -------
    np.ndarray
    """
    if target_fov <= 0 or current_fov <= 0:
        raise ValueError("FoV должен быть > 0°.")

    h,  w  = img.shape[:2]
    cx, cy = center if center else (w // 2, h // 2)

    ratio = target_fov / current_fov          # < 1 → зум‑ин, > 1 → зум‑аут

    # ---- ZOOM‑IN (кроп) ------------------------------------------------------
    if ratio < 1.0:
        new_w, new_h = int(w * ratio), int(h * ratio)

        x0 = np.clip(cx - new_w // 2, 0, w - new_w)
        y0 = np.clip(cy - new_h // 2, 0, h - new_h)
        crop = img[y0:y0 + new_h, x0:x0 + new_w]

        if keep_resolution:
            crop = cv2.resize(crop, (w, h), interpolation=interpolation)
        return crop

    # ---- ZOOM‑OUT (паддинг) --------------------------------------------------
    shrink_w, shrink_h = int(w / ratio), int(h / ratio)
    shrink = cv2.resize(img, (shrink_w, shrink_h), interpolation=interpolation)

    if not keep_resolution:
        return shrink

    # Вкладываем сжатое изображение в «рамку» исходного размера
    canvas = np.full_like(img, border_value)
    x0 = (w - shrink_w) // 2
    y0 = (h - shrink_h) // 2
    canvas[y0:y0 + shrink_h, x0:x0 + shrink_w] = shrink
    return canvas



MODEL_PATH = "unet_trained_on_mixed_v1.h5"
model = load_model(MODEL_PATH, custom_objects={
    'multiclass_dice_loss': multiclass_dice_loss,
    'multiclass_dice_coef': multiclass_dice_coef
})

# Предсказание и сохранение результата
y_pred = model.predict(X)
y_pred_labels = np.argmax(y_pred, axis=-1)

plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(X[50, :, :, 0], cmap="gray")
plt.subplot(1, 5, 2)
plt.imshow(y_pred[50, :, :, 0], cmap="gray")
plt.subplot(1, 5, 3)
plt.imshow(y_pred[50, :, :, 1], cmap="gray")
plt.subplot(1, 5, 4)
plt.imshow(y_pred[50, :, :, 2], cmap="gray")
plt.subplot(1, 5, 5)
plt.imshow(y_pred[50, :, :, 3], cmap="gray")
plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(X[50, :, :, 0], cmap="gray")
plt.subplot(1, 5, 2)
plt.imshow(y[50, :, :, 0], cmap="gray")
plt.subplot(1, 5, 3)
plt.imshow(y[50, :, :, 1], cmap="gray")
plt.subplot(1, 5, 4)
plt.imshow(y[50, :, :, 2], cmap="gray")
plt.subplot(1, 5, 5)
plt.imshow(y[50, :, :, 3], cmap="gray")
plt.show()

y_pred = model.predict(real_data_ready)
# y_pred_labels = np.argmax(y_pred, axis=-1)

plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(real_data_ready[120, :, :, 0], cmap="gray")
plt.subplot(1, 5, 2)
plt.imshow(y_pred[120, :, :, 0], cmap="gray")
plt.subplot(1, 5, 3)
plt.imshow(y_pred[120, :, :, 1], cmap="gray")
plt.subplot(1, 5, 4)
plt.imshow(y_pred[120, :, :, 2], cmap="gray")
plt.subplot(1, 5, 5)
plt.imshow(y_pred[120, :, :, 3], cmap="gray")
plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(real_data_ready[120, :, :, 0], cmap="gray")
plt.subplot(1, 5, 2)
plt.imshow(real_masks_ready[120, :, :, 0], cmap="gray")
plt.subplot(1, 5, 3)
plt.imshow(real_masks_ready[120, :, :, 1], cmap="gray")
plt.subplot(1, 5, 4)
plt.imshow(real_masks_ready[120, :, :, 2], cmap="gray")
plt.subplot(1, 5, 5)
plt.imshow(real_masks_ready[120, :, :, 3], cmap="gray")
plt.show()

# X_list = []
# y_list = []
#
# for pid in patient_ids:
#     t1_path = os.path.join('/training', 'training', f'{pid}', 'pre', 'reg_T1.nii.gz')
#     mask_path = os.path.join('/training', 'training', f'{pid}', 'segm.nii.gz')
#
#     volume = load_nifti(t1_path)
#     mask = load_nifti(mask_path)
#
#     X_pid, y_pid = preprocess_multiclass(volume, mask, mapping=mapping)
#     X_list.append(X_pid)
#     y_list.append(y_pid)
#
# # Объединение всех пациентов в общий массив
# X = np.concatenate(X_list, axis=0)
# y = np.concatenate(y_list, axis=0)
#
# y = y[..., 1:]  # убираем "фон", оставляем только CSF, GM, WM
