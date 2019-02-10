# CNN

Here is some practice with image recognition using the Convolutional Neural Network (CNN) architecture
along with the Canadian Institute For Advanced Research (CIFAR10) image set. This data set consists of 
60,000, 32x32 colored images with labels of the following 10 classes:

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
    
The model consists of 2 convolutional layers, one flattened layer which is then fed into a feed forward network
(Multi-layered perceptron) in the fully connected layer. Max pooling down-sampling operations are used to reduce 
the dimensionality of the feature maps generated from the rectified linear unit (ReLU) activation function.
