import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def multiclass_dice_coef(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true + y_pred, axis=[0, 1, 2])
    dice = K.mean((2. * intersection + smooth) / (union + smooth))
    return dice