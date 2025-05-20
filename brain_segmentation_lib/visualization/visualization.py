import matplotlib.pyplot as plt

class Visualizer:

    @staticmethod
    def plot_slices_and_masks(images, masks, num_slices=5):
        for i in range(min(num_slices, len(images))):
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title(f'Slice {i}')
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f'Mask {i}')
            plt.imshow(masks[i].squeeze(), cmap='gray')
            plt.axis('off')

            plt.show()

    @staticmethod
    def plot_predictions(image, mask_true, mask_pred):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(mask_true.squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Prediction')
        plt.imshow(mask_pred.squeeze(), cmap='gray')
        plt.axis('off')

        plt.show()

    @staticmethod
    def plot_multiclass_mask(image, mask, class_names=["CSF", "GM", "WM"]):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 5, 1)
        plt.title("Image")
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')

        for i, name in enumerate(class_names, 1):
            plt.subplot(1, 5, i + 1)
            plt.title(f"{name} ({i})")
            plt.imshow(mask.squeeze() == i, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
