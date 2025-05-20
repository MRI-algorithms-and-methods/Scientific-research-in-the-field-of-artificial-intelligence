import numpy as np

class ModelEvaluator:
    @staticmethod
    def calculate_average_dice(model, images, masks):
        dices = []
        for img, mask in zip(images, masks):
            pred = model.predict(img[np.newaxis, ...])[0]
            dice = np.mean(2 * (mask * pred) / (mask + pred + 1e-5))
            dices.append(dice)
        return np.mean(dices)