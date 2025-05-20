import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
test_masks_cat = to_categorical(test_masks.squeeze(), num_classes=4)



# Загрузка обученной модели (если ты сохранил)
from tensorflow.keras.models import load_model

model = load_model("unet_trained_on_mixed_v1.h5", custom_objects={
    'multiclass_dice_loss': multiclass_dice_loss,
    'multiclass_dice_coef': multiclass_dice_coef
})

# Предсказания
test_preds = model.predict(test_images)
# new version for deeper network
# pred_labels = np.argmax(model.predict(x), axis=-1)
test_preds_labels = np.argmax(test_preds, axis=-1)

pred_labels_fixed = test_preds_labels.copy()
pred_labels_fixed[(test_preds_labels == 0) & (test_masks.squeeze() == 1)] = 1
pred_labels_fixed[(test_preds_labels == 1) & (test_masks.squeeze() == 0)] = 0



# Визуализация
for i in range(50,52):
    Visualizer.plot_multiclass_mask(test_images[i], pred_labels_fixed[i], class_names=["CSF", "GM", "WM"])
    Visualizer.plot_multiclass_mask(test_images[i], test_masks[i], class_names=["CSF", "GM", "WM"])
