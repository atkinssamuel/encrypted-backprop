from src.constants import Directories, Files
from os import path

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
import shutil


def train_model(model, name, training_data, validation_data=None, batch_size=1, epoch_count=1, learning_rate=0.01,
                checkpoint_frequency=5):
    # creating results folder if it doesn't exist
    if not path.exists(Directories.plaintext_results):
        os.mkdir(Directories.plaintext_results)


    iterations, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    min_valid_loss = None

    # training
    current_iteration = 0  # the number of iterations

    num_batches = math.ceil(training_data.__len__() / batch_size)

    for epoch in range(epoch_count):
        batch_number = 0
        for batch in np.array_split(training_data, num_batches):
            inputs = batch.drop("Class", axis=1).values
            labels = batch["Class"].values

            outputs = model.single_sample_forward(inputs[0, :].reshape(1, -1))
            print("Hello World")
    print("Hello World")