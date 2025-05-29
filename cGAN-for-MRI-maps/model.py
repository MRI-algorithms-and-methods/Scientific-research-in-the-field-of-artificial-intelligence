from tensorflow.keras import layers, models, optimizers, initializers

# PatchGAN discriminator
def define_discriminator(image_shape=(128, 128, 3)):
    init = initializers.RandomNormal(stddev=0.02)

    in_src_image = layers.Input(shape=image_shape)
    in_target_image = layers.Input(shape=image_shape)

    merged = layers.Concatenate()([in_src_image, in_target_image])

    d = layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = layers.Activation('sigmoid')(d)

    model = models.Model([in_src_image, in_target_image], patch_out)

    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = layers.BatchNormalization()(g, training=True)
    g = layers.LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = layers.BatchNormalization()(g, training=True)
    if dropout:
        g = layers.Dropout(0.5)(g, training=True)
    g = layers.Concatenate()([g, skip_in])
    g = layers.Activation('relu')(g)
    return g

# U-Net generator
def define_generator(image_shape=(128, 128, 3)):
    init = initializers.RandomNormal(stddev=0.02)
    in_image = layers.Input(shape=image_shape)

    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)

    b = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
    b = layers.Activation('relu')(b)

    d1 = decoder_block(b, e6, 512)
    d2 = decoder_block(d1, e5, 512)
    d3 = decoder_block(d2, e4, 512, dropout=False)
    d4 = decoder_block(d3, e3, 256, dropout=False)
    d5 = decoder_block(d4, e2, 128, dropout=False)
    d6 = decoder_block(d5, e1, 64, dropout=False)

    g = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6)

    out_image = layers.Activation('tanh')(g)

    model = models.Model(in_image, out_image)
    return model

# cGAN model
def define_gan(g_model, d_model, image_shape=(128, 128, 3)):
    for layer in d_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    in_src = layers.Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = models.Model(in_src, [dis_out, gen_out])
    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model