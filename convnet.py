import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

'''
This is a simple CNN to practice one of the common architectures for image recognition
on the Canadian Institute For Advanced Research (CIFAR10) image data set with 60,000 
32x32 colored images with 10 class labels on:

    1)  Dogs
    2)  Cats
    3)  Horses
    4)  Birds
    5)  Frogs
    6)  Deers
    7)  Ships
    8)  Airplanes
    9)  Trucks
    10) Automobiles
'''


def unpickle(filename):
    '''
    Loads the data from the CIFAR10 dataset into a dictionary.

    :param filename: The relative path of the CIFAR10 image dataset
    :return: The dictionary of encoded image data
    '''

    with open(filename, 'rb') as filey:
        cifar_dict = pickle.load(file=filey, encoding="bytes")
    return cifar_dict


def load_data(data_path):
    '''
    Loads in the data from the CIFAR10 image dataset into the appropriate data batches and test data.

    :param data_path: The relative path of the CIFAR10 image dataset
    :return: The 5 data batches and the test batch
    '''

    # names of data batch files from the CIFAR10 data set
    dirs = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3",
            "data_batch_4", "data_batch_5", "test_batch"]
    all_data = [0, 1, 2, 3, 4, 5, 6]

    # load in the data batches and test batch
    for i, direc in zip(all_data, dirs):
        print("[*] Loading data from " + data_path + direc + " ...")
        all_data[i] = unpickle(data_path + direc)

    data_batch1 = all_data[1]
    data_batch2 = all_data[2]
    data_batch3 = all_data[3]
    data_batch4 = all_data[4]
    data_batch5 = all_data[5]
    test_batch = all_data[6]

    return data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch


def plot_data():
    '''
    Simple function to visualize some of the images.

    :return: Void
    '''

    data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch = load_data("cifar-10-batches-py/")
    X = data_batch1[b"data"]

    # 32x32 sized image with color (hence the 3 for RGB 3 bits of color),
    # transposed to put the color factor as the last arg like usual.
    # uint8 to preserve RAM -> these can be at most 255 or one byte
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

    # Plot some of the images for visualization
    for i in range(10):
        plt.imshow(X[i])
        plt.show()


def one_hot_encoder(vec, vals=10):
    '''
    Encoding the class labels

    :param vec: The data vector
    :param vals: The class labels
    :return: The one-hot vector
    '''

    n = len(vec)
    one_hot = np.zeros((n, vals))
    one_hot[range(n), vec] = 1
    return one_hot


class CNNHelper(object):

    def __init__(self):
        data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch = load_data("cifar-10-batches-py/")

        self.i = 0  # Pointer to the location of the next training batch
        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None

        print("[*] Setting up training images and training labels")
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        # Set training images and normalize pixel values
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encoder(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("[*] Setting up test images and test labels")
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        # Set test images and normalize pixel values
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.test_labels = one_hot_encoder(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def get_batch(self, batch_size):
        '''
        Gets a batch of training data and increments the pointer to the current batch

        :param batch_size: The size of each training batch.
        :return: x - the training images
                 y - the corresponding training labels
        '''

        x = self.training_images[self.i: self.i+batch_size].reshape(10000, 32, 32, 3)
        y = self.training_labels[self.i: self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# plot_data()
helper = CNNHelper()
