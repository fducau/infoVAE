import numpy as np
from utils import mkdir_p
from tensorflow.examples.tutorials import mnist
import os
import random


class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class BasicPropDataset(object):
    def __init__(self):
        data_directory = "BASICPROP"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleNoiseDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle-noise"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleNoiseBGDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle-noise-bg"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

class CarsDataset(object):
    def __init__(self):
        data_directory = "CARS"
        self.dataset_name = data_directory
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.image_dim = 128 * 128
        self.image_shape = (128, 1, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def try_data():
    mnist = MnistDataset()
    basicprop = BasicPropDataset()

def load_data():
    dev_file = 'MNIST/t10k-images-idx3-ubyte'
    dev_data = convert_from_file(dev_file)

    dev_file = 'MNIST/t10k-labels-idx1-ubyte'
    dev_labels = convert_from_file(dev_file)

    train_file = 'MNIST/train-images-idx3-ubyte'
    train_data = convert_from_file(train_file)

    train_file = 'MNIST/train-labels-idx1-ubyte'
    train_labels = convert_from_file(train_file)

def create_data(shuffle=True):
    """ TODO:
        - [x] Shuffle the data
        - [x] Randomize the pixels within some range
    """

    from idx2numpy import convert_to_file

    width = 4

    train_data = []
    train_labels = []
    train_size = 10000
    for i in range(10):
        x = np.zeros((train_size,28,28), dtype=np.uint8)
        offset = i * 2
        line = np.random.randint(150, 240, (train_size,28,width))
        x[:,:,(offset+3):(offset+3+width)] = line
        train_data.append(x)
        y = np.zeros((train_size,), dtype=np.uint8)
        y[:] = i
        train_labels.append(y)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    if shuffle:
        perm = range(train_data.shape[0])
        random.shuffle(perm)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

    eval_data = []
    eval_labels = []
    eval_size = 1000
    for i in range(10):
        x = np.zeros((eval_size,28,28), dtype=np.uint8)
        offset = i * 2
        line = np.random.randint(150, 240, (eval_size,28,width))
        x[:,:,(offset+3):(offset+3+width)] = line
        eval_data.append(x)
        y = np.zeros((eval_size,), dtype=np.uint8)
        y[:] = i
        eval_labels.append(y)

    eval_data = np.concatenate(eval_data, axis=0)
    eval_labels = np.concatenate(eval_labels, axis=0)

    if shuffle:
        perm = range(eval_data.shape[0])
        random.shuffle(perm)
        eval_data = eval_data[perm]
        eval_labels = eval_labels[perm]

    mkdir_p('BASICPROP')
    convert_to_file('BASICPROP/t10k-images-idx3-ubyte', eval_data)
    convert_to_file('BASICPROP/t10k-labels-idx1-ubyte', eval_labels)
    convert_to_file('BASICPROP/train-images-idx3-ubyte', train_data)
    convert_to_file('BASICPROP/train-labels-idx1-ubyte', train_labels)


if __name__ == '__main__':

    pass
    