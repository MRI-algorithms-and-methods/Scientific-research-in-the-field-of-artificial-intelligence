import numpy as np
from skimage.transform import resize


class DataProcessor:
    @staticmethod
    def preprocess_image(img, target_size=(128, 128)):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]  # удаляем лишний канал перед обработкой
        img_resized = resize(img, target_size, preserve_range=True, anti_aliasing=True)
        img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
        return img_normalized[..., np.newaxis]

    @staticmethod
    def binarize_masks(masks, threshold=0.5):
        """Бинаризация масок."""
        return (masks > threshold).astype(np.float32)

    @staticmethod
    def crop_center(img, crop_size=(128, 128)):
        """Обрезка центральной части изображения."""
        y, x = img.shape[:2]
        start_x = x // 2 - crop_size[1] // 2
        start_y = y // 2 - crop_size[0] // 2
        return img[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]
