from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose,
    Concatenate, Dropout
)

def build_unet(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    # Bottleneck
    bn = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    bn = Dropout(0.5)(bn)
    bn = Conv2D(1024, 3, activation='relu', padding='same')(bn)

    # Decoder
    u1 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(bn)
    u1 = Concatenate()([u1, c4])
    c5 = Conv2D(512, 3, activation='relu', padding='same')(u1)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c5)
    u2 = Concatenate()([u2, c3])
    c6 = Conv2D(256, 3, activation='relu', padding='same')(u2)
    c6 = Conv2D(256, 3, activation='relu', padding='same')(c6)

    u3 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c6)
    u3 = Concatenate()([u3, c2])
    c7 = Conv2D(128, 3, activation='relu', padding='same')(u3)
    c7 = Conv2D(128, 3, activation='relu', padding='same')(c7)

    u4 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c7)
    u4 = Concatenate()([u4, c1])
    c8 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    c8 = Conv2D(64, 3, activation='relu', padding='same')(c8)

    outputs = Conv2D(1, 1, activation='sigmoid')(c8)

    return Model(inputs, outputs)


def build_multiclass_unet(input_shape=(128, 128, 1), num_classes=3):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    # Bottleneck
    bn = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    bn = Dropout(0.5)(bn)
    bn = Conv2D(1024, 3, activation='relu', padding='same')(bn)

    # Decoder
    u1 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(bn)
    u1 = Concatenate()([u1, c4])
    c5 = Conv2D(512, 3, activation='relu', padding='same')(u1)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c5)
    u2 = Concatenate()([u2, c3])
    c6 = Conv2D(256, 3, activation='relu', padding='same')(u2)
    c6 = Conv2D(256, 3, activation='relu', padding='same')(c6)

    u3 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c6)
    u3 = Concatenate()([u3, c2])
    c7 = Conv2D(128, 3, activation='relu', padding='same')(u3)
    c7 = Conv2D(128, 3, activation='relu', padding='same')(c7)

    u4 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c7)
    u4 = Concatenate()([u4, c1])
    c8 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    c8 = Conv2D(64, 3, activation='relu', padding='same')(c8)

    outputs = Conv2D(num_classes, 1, activation='softmax')(c8)

    return Model(inputs, outputs)