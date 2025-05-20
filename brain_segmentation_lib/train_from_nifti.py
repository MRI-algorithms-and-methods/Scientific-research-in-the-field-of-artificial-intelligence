import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import tensorflow.keras.backend as K
from brain_segmentation_lib.models.unet import build_multiclass_unet
import albumentations as A
from brain_segmentation_lib.datasets.datasets import PatientDataLoader
from brain_segmentation_lib.visualization.visualization import Visualizer

def load_nifti(path):
    return nib.load(path).get_fdata()

def get_augmentation():
    return A.Compose([
        A.RandomRotate90(p=1.0),
    ])
num_augments = 5

def remap_mask_to_classes(mask, mapping):
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for original_label, new_label in mapping.items():
        new_mask[mask == original_label] = new_label
    return new_mask

def preprocess_multiclass(volume, mask, target_size=(128, 128), mapping=None, augment=False):
    X = []
    y = []
    aug = get_augmentation() if augment else None

    for i in range(volume.shape[2]):
        slice_img = np.rot90(volume[:, :, i], k=-1)
        slice_mask = np.rot90(mask[:, :, i], k=-1)

        if mapping:
            slice_mask = remap_mask_to_classes(slice_mask, mapping)

        if augment:
            for _ in range(num_augments):
                augmented = aug(image=slice_img, mask=slice_mask)
                img_aug = augmented['image']
                mask_aug = augmented['mask']

                # Resize и нормализация
                img_resized = resize(img_aug, target_size, preserve_range=True, anti_aliasing=True)
                mask_resized = resize(mask_aug, target_size, preserve_range=True, anti_aliasing=False, order=0)
                img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))

                X.append(img_norm[..., np.newaxis])
                y.append(mask_resized.astype(np.uint8))
        else:
            # Только оригинал
            slice_img_resized = resize(slice_img, target_size, preserve_range=True, anti_aliasing=True)
            slice_mask_resized = resize(slice_mask, target_size, preserve_range=True, anti_aliasing=False, order=0)
            slice_img_norm = (slice_img_resized - np.min(slice_img_resized)) / (
                        np.max(slice_img_resized) - np.min(slice_img_resized))

            X.append(slice_img_norm[..., np.newaxis])
            y.append(slice_mask_resized.astype(np.uint8))

        slice_img_resized = resize(slice_img, target_size, preserve_range=True, anti_aliasing=True)
        slice_mask_resized = resize(slice_mask, target_size, preserve_range=True, anti_aliasing=False, order=0)

        slice_img_norm = (slice_img_resized - np.min(slice_img_resized)) / (np.max(slice_img_resized) - np.min(slice_img_resized))

        X.append(slice_img_norm[..., np.newaxis])
        y.append(slice_mask_resized.astype(np.uint8))

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=4)
    return X, y


def multiclass_dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    union = K.sum(y_true + y_pred, axis=[0,1,2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)

def multiclass_dice_loss(y_true, y_pred):
    return 1 - multiclass_dice_coef(y_true, y_pred)


# Список пациентов для обучения
patient_ids = [1,4,5,7,14,27]

X_all = []
y_all = []

mapping = {
    5: 1, 6: 1,
    1: 2, 2: 2,
    3: 3, 4: 3
}



for pid in patient_ids:
    t1_path = os.path.join('F:/model/ml/training', 'training', f'{pid}', 'pre', 'reg_T1.nii.gz')
    mask_path = os.path.join('F:/model/ml/training', 'training', f'{pid}', 'segm.nii.gz')

    if os.path.exists(t1_path) and os.path.exists(mask_path):
        print(f"Загрузка пациента {pid}")
        volume = load_nifti(t1_path)
        mask = load_nifti(mask_path)
        X_pid, y_pid = preprocess_multiclass(volume, mask, mapping=mapping, augment=True)
        X_all.append(X_pid)
        y_all.append(y_pid)
    else:
        print(f"Пропущен пациент {pid}, файл не найден.")

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
print("Форма X:", X.shape, "Форма y:", y.shape)

# model = build_multiclass_unet(input_shape=(128, 128, 1), num_classes=4)
from tensorflow.keras.models import load_model

MODEL_PATH = "unet_finetuned_first_real_then_synth_v2.h5"
model = load_model(MODEL_PATH, custom_objects={
    'multiclass_dice_loss': multiclass_dice_loss,
    'multiclass_dice_coef': multiclass_dice_coef
})
import numpy as np

def random_subset(X, y, percent=0.3, seed=None):
    """
    Возвращает случайный поднабор данных X и y в заданном проценте.

    :param X: np.array, shape (N, ...)
    :param y: np.array, shape (N, ...)
    :param percent: float, доля от исходного массива (например, 0.3 для 30%)
    :param seed: int, опционально, для воспроизводимости
    :return: X_subset, y_subset
    """
    assert 0 < percent <= 1, "Процент должен быть в пределах (0, 1]"
    n = len(X)
    k = int(n * percent)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)

    return X[indices], y[indices]


X_sub, y_sub = random_subset(X, y, percent=0.3)


model.compile(optimizer=Adam(1e-4), loss=multiclass_dice_loss, metrics=[multiclass_dice_coef])

history = model.fit(X_sub, y_sub, batch_size=8, epochs=15, validation_split=0.1)

model.save("unet_trained_on_nifti_update.h5")
# графики
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['multiclass_dice_coef'], label='Train Dice')
plt.plot(history.history['val_multiclass_dice_coef'], label='Val Dice')
plt.title("Коэффициент Дайса. Обучение на реальных данных")
plt.xlabel("Эпоха")
plt.ylabel("Dice")
plt.legend()
plt.grid()
plt.savefig("history_realdata_50_ep_finetuned_twice.png")
plt.show()
