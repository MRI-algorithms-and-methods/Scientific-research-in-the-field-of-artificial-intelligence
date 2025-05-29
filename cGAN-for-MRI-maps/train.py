import numpy as np
import utils
import model

# train models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset

    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    print_steps = max(1, bat_per_epo // 10)

    for i in range(n_steps):
        [X_realA, X_realB], y_real = utils.generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = utils.generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        if (i + 1) % print_steps == 0 or (i + 1) == n_steps:
            epoch = (i + 1) // bat_per_epo + 1
            step_in_epoch = (i + 1) % bat_per_epo
            print(f'Epoch {epoch}, Step {step_in_epoch}/{bat_per_epo}, '
                  f'd1[{d_loss1:.3f}] d2[{d_loss2:.3f}] g[{g_loss:.3f}]')

        if (i + 1) % (bat_per_epo*5) == 0:
            utils.summarize_performance(i, g_model, dataset)

# load data
dataset = np.load('data/mri_dataset_all.npz')
dataset = [dataset['X1'], dataset['X2']]

# define input shape
image_shape = dataset[0].shape[1:]

# define models
d_model = model.define_discriminator(image_shape)
g_model = model.define_generator(image_shape)
gan_model = model.define_gan(g_model, d_model, image_shape)

# train models
train(d_model, g_model, gan_model, dataset, n_epochs=50, n_batch=1)