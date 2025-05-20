
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

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
        slice_img = np.rot90(volume[:, :, i], k=-1)     # Поворот изображения
        slice_mask = np.rot90(mask[:, :, i], k=-1)      # Поворот маски

        if mapping:
            slice_mask = remap_mask_to_classes(slice_mask, mapping)

        slice_img_resized = resize(slice_img, target_size, preserve_range=True, anti_aliasing=True)
        slice_mask_resized = resize(slice_mask, target_size, preserve_range=True, anti_aliasing=False, order=0)

        slice_img_norm = (slice_img_resized - np.min(slice_img_resized)) / (np.max(slice_img_resized) - np.min(slice_img_resized))

        X.append(slice_img_norm[..., np.newaxis])
        y.append(slice_mask_resized.astype(np.uint8))

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=4)  # 0 - фон, 1-3 - классы
    return X, y

def plot_multiclass_prediction(image, true_mask, pred_mask, class_names=["BG", "CSF", "GM", "WM"]):
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
    plt.show()

# Метрики для загрузки кастомной модели
def multiclass_dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    union = K.sum(y_true + y_pred, axis=[0,1,2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)

def multiclass_dice_loss(y_true, y_pred):
    return 1 - multiclass_dice_coef(y_true, y_pred)

# Конфигурация
patient_test = 29
T1_PATH = os.path.join('F:/model/ml/training', 'training', f'{patient_test}', 'pre', 'reg_T1.nii.gz')
MASK_PATH = os.path.join('F:/model/ml/training', 'training', f'{patient_test}', 'segm.nii.gz')
MODEL_PATH = 'unet_finetuned_nifti.h5'  # модель с num_classes=4

# Маппинг маски
mapping = {
    5: 1, 6: 1,      # CSF
    1: 2, 2: 2,      # GM
    3: 3, 4: 3       # WM
}

# Загрузка и предобработка
volume = load_nifti(T1_PATH)
mask = load_nifti(MASK_PATH)
X_test, y_test_cat = preprocess_multiclass(volume, mask, mapping=mapping)

print("X_test:", X_test.shape, "| y_test_cat:", y_test_cat.shape)

# Загрузка модели
model = load_model(MODEL_PATH, custom_objects={
    'multiclass_dice_loss': multiclass_dice_loss,
    'multiclass_dice_coef': multiclass_dice_coef
})

# Предсказания
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=-1)+ 1

loc_range = range(27,28)

# Визуализация
for i in loc_range:
    plot_multiclass_prediction(X_test[i], np.argmax(y_test_cat[i], axis=-1)+1, y_pred_labels[i])


def plot_ground_truth(image, true_mask, class_names=["BG", "CSF", "GM", "WM"]):
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("T1 Image")
    plt.axis("off")

    for i in range(1, 4):
        plt.subplot(1, 4, i + 1)
        plt.imshow((true_mask == i).astype(np.uint8), cmap="gray")
        plt.title(f"GT {class_names[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Отображение Ground Truth масок
for i in loc_range:
    print(f"Slice {i} (Ground Truth)")
    plot_ground_truth(X_test[i], np.argmax(y_test_cat[i], axis=-1))



from brain_segmentation_lib.datasets.datasets import PatientDataLoader
from brain_segmentation_lib.visualization.visualization import Visualizer
from brain_segmentation_lib.models.unet import build_multiclass_unet
from brain_segmentation_lib.utils.losses import multiclass_dice_loss
from brain_segmentation_lib.utils.metrics import multiclass_dice_coef


test_masks_dir = 'F:/model/ml/testing/test_anatomic_masks'
testing_t1_dir = 'F:/model/ml/testing/test_weighted_t1'
# 🔍 Проверка модели на новом пациенте (18)
patient_id = 18

# Загрузка тестового пациента
test_images, test_masks = PatientDataLoader.load_multiclass_patient_from_t1(testing_t1_dir, test_masks_dir, patient_id)
test_masks_cat = to_categorical(test_masks.squeeze(), num_classes=4)[..., 1:]

# Предсказания
test_preds = model.predict(test_images)
# new version for deeper network
# pred_labels = np.argmax(model.predict(x), axis=-1)
test_preds_labels = np.argmax(test_preds, axis=-1) + 1

pred_labels_fixed = test_preds_labels.copy()
pred_labels_fixed[(test_preds_labels == 0) & (test_masks.squeeze() == 1)] = 1
pred_labels_fixed[(test_preds_labels == 1) & (test_masks.squeeze() == 0)] = 0

# Визуализация
for i in range(60,61):
    Visualizer.plot_multiclass_mask(test_images[i], pred_labels_fixed[i], class_names=["CSF", "GM", "WM"])
    Visualizer.plot_multiclass_mask(test_images[i], test_masks[i], class_names=["CSF", "GM", "WM"])
