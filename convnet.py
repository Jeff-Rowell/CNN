import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

'''
This is a simple CNN to practice one of the common architectures for image recognition
on the Canadian Institute For Advanced Research (CIFAR-10) image data set with 60,000 
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
    dirs = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
    all_data = [0, 1, 2, 3, 4, 5, 6]

    # load in the data batches and test batch
    for i, direc in zip(all_data, dirs):
        print("[*] Loading data from " + data_path + direc + " ...")
        all_data[i] = unpickle(data_path + direc)

    batch_meta = all_data[0]
    data_batch1 = all_data[1]
    data_batch2 = all_data[2]
    data_batch3 = all_data[3]
    data_batch4 = all_data[4]
    data_batch5 = all_data[5]
    test_batch = all_data[6]

    print("\n")
    print(batch_meta)
    return data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch


data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch = load_data("cifar-10-batches-py/")

X = data_batch1[b"data"]
# 32x32 sized image with color (hence the 3 for RGB 3 bits of color), transposed to put the 3 as the last arg as norm
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

