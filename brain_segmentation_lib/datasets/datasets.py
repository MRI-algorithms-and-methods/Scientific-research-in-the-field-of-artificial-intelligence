import os
import re
import numpy as np
import nibabel as nib
from skimage.transform import resize

class PatientDataLoader:

    # @staticmethod
    # def map_labels_to_3classes(mask):
    #     new_mask = np.zeros_like(mask, dtype=np.uint8)
    #     new_mask[np.isin(mask, [5, 6])] = 1  # CSF
    #     new_mask[np.isin(mask, [1, 2])] = 2  # GM
    #     new_mask[np.isin(mask, [3, 4])] = 3  # WM
    #     return new_mask

    @staticmethod
    def load_multiclass_patient_from_t1(t1_dir, masks_dir, patient_id, target_size=(128, 128)):
        import re
        from skimage.transform import resize

        pattern = re.compile(rf"subject{patient_id:02d}_crisp_v_(\d+)\.npy")

        images = []
        masks = []

        for file in os.listdir(t1_dir):
            match = pattern.match(file)
            if not match:
                continue

            slice_id = int(match.group(1))
            t1_path = os.path.join(t1_dir, file)
            mask_path = os.path.join(masks_dir, file)

            if not os.path.exists(mask_path):
                continue

            # Загрузка
            image_slice = np.load(t1_path)
            mask = np.load(mask_path)

            # Преобразование маски
            mapped_mask = PatientDataLoader.map_labels_to_3classes(mask)

            # Resize
            image_resized = resize(image_slice, target_size, preserve_range=True)
            mask_resized = resize(mapped_mask, target_size, preserve_range=True, order=0, anti_aliasing=False)

            # Нормализация
            image_norm = (image_resized - np.min(image_resized)) / (np.max(image_resized) - np.min(image_resized))

            images.append(image_norm[..., np.newaxis])
            masks.append(mask_resized.astype(np.uint8)[..., np.newaxis])

        return np.array(images), np.array(masks)

    @staticmethod
    def map_labels_to_3classes(mask):
        """
        Преобразует маску с метками от 0 до 9 в маску с 3 классами:
        1 - CSF, 2 - GM, 3 - WM. Остальные игнорируются (становятся 0).
        """
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask[mask == 1] = 1  # CSF
        new_mask[mask == 2] = 2  # GM
        new_mask[mask == 3] = 3  # WM
        return new_mask

    @staticmethod
    def load_patient_volume(images_dir, patient_id):
        image_file = f"subject{patient_id:02d}_crisp_v.mnc"
        image_path = os.path.join(images_dir, image_file)
        img = nib.load(image_path)
        return img.get_fdata()

    @staticmethod
    def load_mask_stack(masks_dir, patient_id):
        masks = []
        slices = []

        pattern = re.compile(rf"subject{patient_id:02d}_crisp_v_(\d+)\.npy")

        for f in os.listdir(masks_dir):
            match = pattern.match(f)
            if match:
                slice_id = int(match.group(1))
                mask = np.load(os.path.join(masks_dir, f))
                masks.append(mask)
                slices.append(slice_id)

        return slices, masks

    @staticmethod
    def prepare_patient_data(volume, slices, raw_masks, target_size=(128, 128)):
        images = []
        masks = []

        for slice_id, raw_mask in zip(slices, raw_masks):
            image_slice = volume[slice_id, :, :]
            mapped_mask = PatientDataLoader.map_labels_to_3classes(raw_mask)

            image_resized = resize(image_slice, target_size, preserve_range=True)
            mask_resized = resize(mapped_mask, target_size, preserve_range=True, order=0, anti_aliasing=False)

            image_norm = (image_resized - np.min(image_resized)) / (np.max(image_resized) - np.min(image_resized))

            images.append(image_norm[..., np.newaxis])
            masks.append(mask_resized.astype(np.uint8)[..., np.newaxis])

        return np.array(images), np.array(masks)

    @staticmethod
    def load_multiclass_patient(images_dir, masks_dir, patient_id):
        volume = PatientDataLoader.load_patient_volume(images_dir, patient_id)
        slices, raw_masks = PatientDataLoader.load_mask_stack(masks_dir, patient_id)
        return PatientDataLoader.prepare_patient_data(volume, slices, raw_masks)