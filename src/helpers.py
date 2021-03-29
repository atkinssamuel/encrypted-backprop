import numpy as np
import math


def xavier_init(r, c, D):
    """
    Returns an r x c matrix with initialization scaling factor D
    :param r: number of rows
    :param c: number of columns
    :param D: initialization scaling factor
    :return: numpy array
    """
    return np.random.randn(r, c) * D


def round_sig(x, sig=2):
    """
    Rounds the input "x" to "sig" significant digits
    :param x:
    :param sig:
    :return:
    """
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)


def ReLU(x):
    """
    Takes in an input matrix of any size and computes the ReLU function element-wise
    :param x: numpy array
    :return: numpy array
    """
    return x * (x > 0)


def sigmoid(x):
    """
    Computes the sigmoid function on the elements in the input matrix
    :param x: Nx1 numpy array
    :return: Nx1 numpy array
    """
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(prediction, label):
    """
    Computes the cross-entropy loss function from the prediction and label inputs
    :param prediction: (N,) np.array of floats
    :param label: (N,) np.array of integers
    :return: (N,) np.array of floats
    """
    return -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))


def element_wise_step(x):
    """
    Computes the step function on the floating input value x
    :param x: float
    :return: integer
    """
    if x == 0:
        return 0
    return x/abs(x)


def step(x):
    """
    Computes the step function on the array input x
    :param x: array of floats
    :return: array of integers
    """
    stepper = np.vectorize(element_wise_step)
    return (stepper(x[0])).reshape(x.shape)


def average(x):
    """
    Computes the column-wise average of the elements in x
    :param x: list
    :return: list
    """
    x = np.array(x)
    return np.average(x, axis=0)


def compute_accuracy(predictions, labels):
    """
    Computes the accuracy
    :param predictions: (N,) np.array of floats
    :param labels: (N,) np.array of integers
    :return:
    """
    correct_mask = np.round(predictions) == labels
    return np.sum(correct_mask)/correct_mask.shape[0]