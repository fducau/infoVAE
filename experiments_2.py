import numpy as np
import datetime
import dateutil
import copy
from misc.datasets import BasicPropDataset, \
                          BasicPropAngleDataset, \
                          BasicPropAngleNoiseDataset, \
                          BasicPropAngleNoiseBGDataset, \
                          MnistDataset
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from vae_half import *


parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', default=20,
                    help='Latent code dimension',
                    type=int)
parser.add_argument('--epochs', default=50, 
                    help='Number of epochs',
                    type=int)

args = parser.parse_args()
latent_dim = args.latent_dim
n_epochs = args.epochs

DATASETS = ['MNIST', 'BASICPROP', 'BPAngle', 'BPAngleNoise', 'BPAngleNoiseBG']
SAVEPLOTS = './plots'


def plot_reconstruction(network_architecture, info=False, dataset='MNIST', x_sample=None):

    # Validate dataset
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    if x_sample is None:
        # Reconstruct test images using noinfo architecture
        x_sample = dataset.test.next_batch(100)[0]
    if dataset.dataset_name == "BASICPROP-angle":
        x_sample = np.ceil(x_sample)

    # Train network
    vae = train(network_architecture, training_epochs=n_epochs,
                info=info, dataset=dataset)

    # Plot reconstructions
    x_reconstruct = vae.reconstruct(x_sample)
    plt.figure(figsize=(8, 15))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        if i == 0:
            plt.title("Original")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        if i == 0:
            plt.title("Reconstruction")
        plt.colorbar()
    plt.suptitle('Info {}'.format(info))

    # Save plot
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%H_%M_%S_%Y%m%d')

    savepath = '{}/REC_DS-{}_nz{}_info{}_{}'.format(SAVEPLOTS,
                                                    dataset.dataset_name,
                                                    network_architecture['n_z'],
                                                    info, timestamp)
    plt.savefig(savepath)


def plots_2D(network_architecture, info=False, dataset='MNIST', x_sample=None):
    # Validate number of dimensions
    if network_architecture['n_z'] > 2:
        network_architecture = copy.deepcopy(network_architecture)
        network_architecture['n_z'] = 2

    # Validate dataset
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    if x_sample is None:
        # Reconstruct test images using noinfo architecture
        x_sample = dataset.test.next_batch(100)[0]
    if dataset.dataset_name == "BASICPROP-angle":
        x_sample = np.ceil(x_sample)

    # Train network
    vae_2d = train(network_architecture, training_epochs=n_epochs,
                   info=info, dataset=dataset)

    # 2D scatterplot
    z_mu = vae_2d.transform(x_sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1])
    plt.title("Latent Space {}".format(info))
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.grid()

    # Save plot
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%H_%M_%S_%Y%m%d')

    savepath = '{}/LAT_nz{}_info{}_{}'.format(SAVEPLOTS,
                                              network_architecture['n_z'],
                                              info, timestamp)
    plt.savefig(savepath)

    # 2D reconstructions
    # X-axis: second dimension, Y-axis: first dimension
    nx = ny = 8
    x_values = np.linspace(-2.5, 2.5, nx)
    y_values = np.linspace(-2.5, 2.5, ny)

    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        ax = plt.subplot(g)
        z_mu = np.array([[x_values[i / ny], y_values[i % nx]]] * 100)
        x_mean = vae_2d.generate(z_mu)
        ax.imshow(x_mean[0].reshape(28, 28))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

    plt.suptitle('Info {}'.format(info))

    # Save plot
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%H_%M_%S_%Y%m%d')

    savepath = '{}/LAT_REC_nz{}_info{}_{}'.format(SAVEPLOTS,
                                                  network_architecture['n_z'],
                                                  info, timestamp)
    plt.savefig(savepath)

def main():

    network_architecture = dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
                                n_hidden_recog_2=500,  # 2nd layer encoder neurons
                                n_hidden_gener_1=500,  # 1st layer decoder neurons
                                n_hidden_gener_2=500,  # 2nd layer decoder neurons
                                n_input=784,  # MNIST data input (img shape: 28*28)
                                n_z=latent_dim,       # dimensionality of latent space
                                info=False)

    for dataset_name in DATASETS:
        for info in [True, False]:
            dataset = load_dataset(dataset_name)
            x_sample = x_sample = dataset.test.next_batch(100)[0]
            plot_reconstruction(network_architecture,
                                info=info,
                                dataset=dataset,
                                x_sample=x_sample)

    for dataset_name in DATASETS:
        for info in [True, False]:
            dataset = load_dataset(dataset_name)
            x_sample, y_sample = dataset.test.next_batch(5000)
            plots_2D(network_architecture,
                     info=info,
                     dataset=dataset,
                     x_sample=x_sample)


if __name__ == '__main__':
    main()

