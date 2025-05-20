from .metrics import dice_coef, multiclass_dice_coef

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)



def multiclass_dice_loss(y_true, y_pred):
    return 1 - multiclass_dice_coef(y_true, y_pred)