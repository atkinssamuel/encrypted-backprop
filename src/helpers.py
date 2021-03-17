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
    :param x: numpy array
    :return: numpy array
    """
    return 1 / (1 + math.exp(-x))
