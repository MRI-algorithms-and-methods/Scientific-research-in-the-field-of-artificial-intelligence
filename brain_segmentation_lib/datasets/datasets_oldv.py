import os
import re
import numpy as np
import nibabel as nib
from skimage.transform import resize


class PatientDataLoader:

    @staticmethod
    def select_patient(images_dir, masks_dir, patient_id):
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.npy')]
        parsed_masks = []

        for mask_file in mask_files:
            match = re.match(r'subject(\d+)_crisp_v_(\d+).npy', mask_file)
            if match and int(match.group(1)) == patient_id:
                parsed_masks.append({
                    'slice_id': int(match.group(2)),
                    'mask_path': os.path.join(masks_dir, mask_file)
                })

        parsed_masks.sort(key=lambda x: x['slice_id'])
        return parsed_masks

    @staticmethod
    def load_patient_data(images_dir, masks_dir, patient_id, target_size=(128, 128)):
        image_file = f'subject{patient_id:02d}_crisp_v.mnc'
        image_path = os.path.join(images_dir, image_file)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Patient image not found: {image_path}")

        img = nib.load(image_path)
        patient_volume = img.get_fdata()

        masks_info = PatientDataLoader.select_patient(images_dir, masks_dir, patient_id)

        images, masks = [], []
        for mask_info in masks_info:
            slice_id = mask_info['slice_id']
            image_slice = patient_volume[slice_id, :, :]
            mask_slice = np.load(mask_info['mask_path'])

            image_resized = resize(image_slice, target_size, preserve_range=True)
            mask_resized = resize(mask_slice, target_size, preserve_range=True, anti_aliasing=False)

            image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())

            images.append(image_normalized[..., np.newaxis])
            masks.append((mask_resized > 0.5).astype(np.float32)[..., np.newaxis])

        return np.array(images), np.array(masks)
