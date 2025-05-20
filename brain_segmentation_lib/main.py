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


from skimage.transform import rotate

def augment_with_rotations(images, masks, num_augments=4, angles_range=(-30, 30)):
    aug_images = []
    aug_masks = []

    for img, msk in zip(images, masks):
        aug_images.append(img)
        aug_masks.append(msk)

        for _ in range(num_augments):
            angle = np.random.uniform(*angles_range)
            img_rot = rotate(img.squeeze(), angle, resize=False, preserve_range=True)
            msk_rot = rotate(msk.squeeze(), angle, resize=False, order=0, preserve_range=True)

            aug_images.append(img_rot[..., np.newaxis])
            aug_masks.append(msk_rot[..., np.newaxis])

    return np.array(aug_images), np.array(aug_masks)


# GPU: включаем "memory growth"
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# параметры
EPOCHS = 50
BATCH_SIZE = 16
# images_dir = 'F:/model/ml/training/mnc'
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

plt.imshow(images[50])


images, masks = augment_with_rotations(images, masks, num_augments=4)
masks_cat = to_categorical(masks.squeeze(), num_classes=4)

print("After augmentation:", images.shape, masks.shape)
print("Images shape:", images.shape)
print("Masks one-hot shape:", masks_cat.shape)

# визуализация ground truth
for i in range(50,52):
    Visualizer.plot_multiclass_mask(images[i], masks[i], class_names=["BG", "CSF", "GM", "WM"])

# сборка и компиляция модели
model = build_multiclass_unet(input_shape=(128, 128, 1), num_classes=4)
model.compile(optimizer=Adam(1e-4), loss=multiclass_dice_loss, metrics=[multiclass_dice_coef])
model.summary()

# обучение
history = model.fit(
    images,
    masks_cat,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# графики
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['multiclass_dice_coef'], label='Train Dice')
plt.plot(history.history['val_multiclass_dice_coef'], label='Val Dice')
plt.title("Коэффициент Дайса")
plt.xlabel("Эпоха")
plt.ylabel("Dice")
plt.legend()
plt.grid()
plt.show()

# сохранение модели
model_filename = f"unet_epochs{EPOCHS}_batch{BATCH_SIZE}.h5"
model.save(model_filename)
print(f"✅ Модель сохранена в: {model_filename}")

# предсказания
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=-1)

# сравнение: GT vs Prediction (рядом)
for i in range(50,53):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(masks[i], cmap='gray')
    axs[0].set_title("Ground Truth")
    axs[1].imshow(pred_labels[i], cmap='gray')
    axs[1].set_title("Prediction")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"Slice {i}")
    plt.tight_layout()
    plt.show()


# 🔍 Проверка модели на новом пациенте (18)
patient_id = 18

# Загрузка тестового пациента
test_images, test_masks = PatientDataLoader.load_multiclass_patient(testing_t1_dir, test_masks_dir, patient_id)
test_masks_cat = to_categorical(test_masks.squeeze(), num_classes=4)[..., 1:]

# Загрузка обученной модели (если ты сохранил)
# from tensorflow.keras.models import load_model
# model = load_model("trained_model_multiclass.h5", custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

# Предсказания
test_preds = model.predict(test_images)
test_preds_labels = np.argmax(test_preds, axis=-1)

# Визуализация
for i in range(50,52):
    Visualizer.plot_multiclass_mask(test_images[i], test_preds_labels[i], class_names=["CSF", "GM", "WM"])
