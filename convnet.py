import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

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

        x = self.training_images[self.i: self.i+batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i: self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# plot_data()
helper = CNNHelper()


class CNN(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # 32x32 colored images with 3 bits for RGB
        self.y_true = tf.placeholder(tf.float32, shape=[None, 10])    # 10 class labels
        self.hold_prob = tf.placeholder(tf.float32)                   # holder for dropout to decrease model bias

    def init_weights(self, shape):
        '''
        Initialize weights using a random distribution and standard deviation of 0.1.

        :param shape: The shape of the weights
        :return: Randomized weights using a standard deviation of 0.1
        '''

        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        '''
        Initialize the bias neuron.

        :param shape: The shape of the bias
        :return: Constant value of 0.1 for the bias.
        '''

        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(self, x, W):
        '''
        Filter the pixel data x with W using size 1 strides.

        :param x: The input pixels matrix
        :param W: The weights matrix/filter matrix
        :return: The convolved filter.
        '''

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2by2(self, x):
        '''
        Max pooling layer to reduce the data down.

        :param x: The convolved feature matrix.
        :return: The reduced max pooled matrix.
        '''

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def convolution_layer(self, input_x, shape):
        '''
        Convolution layer to compute convolved feature matrices.

        :param input_x: The original input pixel matrix.
        :param shape: The shape of the input tensor.
        :return: Rectified feature mapping.
        '''

        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)

    def fully_connected_layer(self, input_layer, size):
        '''
        Typical feed forward neural network.

        :param input_layer: The flattened pooled feature map.
        :param size: The size of the flattened vector.
        :return: Predicted output
        '''

        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b

    def build_layers(self):
        '''
        Builds the CNN model.

        :return: Predicted outputs and the training function.
        '''

        # First convolution layer
        conv1 = self.convolution_layer(self.x, shape=[4, 4, 3, 32])
        conv1_pooling = self.max_pool_2by2(conv1)

        # Second convolution layer
        conv2 = self.convolution_layer(conv1_pooling, shape=[4, 4, 32, 64])
        conv2_pooling = self.max_pool_2by2(conv2)

        # Flattened layer
        conv2_flattened = tf.reshape(conv2_pooling, [-1, 8 * 8 * 64])

        fc_layer1 = tf.nn.relu(self.fully_connected_layer(conv2_flattened, 1024))
        fc_dropout = tf.nn.dropout(fc_layer1, keep_prob=self.hold_prob)
        y_predictions = self.fully_connected_layer(fc_dropout, 10)

        # Loss function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_true,
                                                                               logits=y_predictions))

        # Optimization
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)
        train = opt.minimize(cross_entropy)

        return y_predictions, train

    def train_model(self, train, y_predictions):
        print("\n[+] Training CNN model...\n")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(1250):
                batch = helper.get_batch(100)
                sess.run(train, feed_dict={self.x: batch[0], self.y_true: batch[1], self.hold_prob: 0.5})

                if i % 100 == 0:
                    print("Currently on iteration %d" % i)

                    matches = tf.equal(tf.argmax(y_predictions, 1), tf.argmax(self.y_true, 1))
                    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                    accuracy = sess.run(accuracy, feed_dict={self.x: helper.test_images,
                                                             self.y_true: helper.test_labels,
                                                             self.hold_prob: 1.0})
                    print("Accuracy is: %f\n" % accuracy)
        return accuracy


cnn = CNN()
y_predictions, train = cnn.build_layers()

start = time.time()
accuracy = cnn.train_model(train=train, y_predictions=y_predictions)
elapsed = time.time() - start

print("\n\n[+] Final trained accuracy is: %f\n[+] Time to train: %f seconds" % (accuracy, elapsed))
