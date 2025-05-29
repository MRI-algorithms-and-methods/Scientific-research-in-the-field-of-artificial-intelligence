import numpy as np
import matplotlib.pyplot as plt


# GAN utils

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1), dtype='float32')
    return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1), dtype='float32')
    return X, y


def summarize_performance(step, g_model, dataset, n_samples=1):
    channel_names = ['T1', 'T2', 'PD']

    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    # scale from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    fig, axes = plt.subplots(3, 3, figsize=(8, 6))

    input_img = X_realA[0]
    fake_img = X_fakeB[0]
    real_img = X_realB[0]

    for ch in range(3):
        axes[0, ch].imshow(input_img[:, :, ch])
        axes[0, ch].set_title(f"Input (weighted) - {channel_names[ch]}")
        axes[0, ch].axis('off')

        axes[1, ch].imshow(fake_img[:, :, ch])
        axes[1, ch].set_title(f"Generated maps- {channel_names[ch]}")
        axes[1, ch].axis('off')

        axes[2, ch].imshow(real_img[:, :, ch])
        axes[2, ch].set_title(f"Real maps - {channel_names[ch]}")
        axes[2, ch].axis('off')

    plt.tight_layout()
    filename1 = f'plot_{step+1:06d}.png'
    plt.savefig(filename1)
    plt.show()
    plt.close()

    filename2 = f'model_{step+1:06d}.keras'
    g_model.save(filename2)
    print(f'Saved: {filename1} and {filename2}')
