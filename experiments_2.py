from misc.datasets import BasicPropDataset, \
                          BasicPropAngleDataset, \
                          BasicPropAngleNoiseDataset, \
                          BasicPropAngleNoiseBGDataset

import matplotlib.pyplot as plt
from vae_half import *


def plot_reconstruction()
    pass


def main()

    dataset = BasicPropAngleNoiseDataset()

    vae_noinfo = train(network_architecture, training_epochs=25, info=False, dataset='BPAngleNoise')
    vae_info = train(network_architecture, training_epochs=25, info=True, dataset='BPAngleNoise')

    # Reconstruct test images using noinfo architecture
    x_sample = dataset.test.next_batch(100)[0]

    x_reconstruct = vae_noinfo.reconstruct(x_sample)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("VAE noinfo")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()

    # Reconstruct test images using info architecture
    x_reconstruct = vae_info.reconstruct(x_sample)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("VAE info")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()

    # 2D analysis
    network_architecture_2d = \
        dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
             n_hidden_recog_2=500,  # 2nd layer encoder neurons
             n_hidden_gener_1=500,  # 1st layer decoder neurons
             n_hidden_gener_2=500,  # 2nd layer decoder neurons
             n_input=784,  # MNIST data input (img shape: 28*28)
             n_z=2,        # dimensionality of latent space
             info=False)

    vae_2d_noinfo = train(network_architecture_2d, training_epochs=25, info=False, dataset='BPAngleNoise')
    vae_2d_info = train(network_architecture_2d, training_epochs=25, info=True, dataset='BPAngleNoise')

    x_sample, y_sample = dataset.test.next_batch(5000)

    # 2D scatterplot noinfo
    z_mu = vae_2d_noinfo.transform(x_sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1])
    plt.title("Latent Space noinfo")
    plt.grid()

    # 2D scatterplot info
    z_mu = vae_2d_info.transform(x_sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1])
    plt.title("Latent Space info")
    plt.grid()

    # Latent space reconstruction plots
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    # 2D plot with noinfo
    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * 100)
            x_mean = vae_2d_noinfo.generate(z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    plt.title("Latent Space Reconstructions noinfo")
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()

    # 2D plot with info
    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * 100)
            x_mean = vae_2d_info.generate(z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    plt.title("Latent Space Reconstructions info")
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
if __name__ == '__main__':
    main()


plt.subplot()
gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
for i,g in enumerate(gs):
    ax = plt.subplot(g)
    z_mu = np.array([[x_values[i / 20], y_values[i % 20]]] * 100)
    x_mean = vae_2d_info.generate(z_mu)
    ax.imshow(x_mean[0].reshape(28, 28))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')


plt.subplot()
gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
for i,g in enumerate(gs):
    ax = plt.subplot(g)
    z_mu = np.array([[x_values[i / 20], y_values[i % 20]]] * 100)
    x_mean = vae_2d_noinfo.generate(z_mu)
    ax.imshow(x_mean[0].reshape(28, 28))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
