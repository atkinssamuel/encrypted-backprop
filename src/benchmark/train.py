from src.benchmark.helpers import get_accuracy, get_loss
from src.constants import Directories
from os import path

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal

def train_model(model, name, training_data, validation_data=None, batch_size=1, epoch_count=1, shuffle=False,
          learning_rate=0.01, checkpoint_frequency=5, momentum=0.9):

    if not path.exists(Directories.benchmark_models):
        os.mkdir(Directories.benchmark_models)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iterations, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    # training
    current_iteration = 0  # the number of iterations

    for epoch in range(epoch_count):
        for batch in np.array_split(training_data, math.ceil(training_data.__len__()/batch_size)):
            inputs = torch.tensor(batch.drop("Class", axis=1).values).reshape(shape=(-1, 1, 30))
            labels = torch.tensor(batch["Class"].values).reshape(shape=(-1, 1, 1))
            optimizer.zero_grad()  # a clean up step for PyTorch
            outputs = model(inputs.float())  # forward pass
            loss = criterion(outputs, labels.float())  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter

            # save the current training information
            iterations.append(current_iteration)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            train_acc.append(get_accuracy(model, training_data))  # compute training accuracy

            if validation_data is not None:
                validation_acc.append(get_accuracy(model, validation_data))
                validation_loss.append(get_loss(model, validation_data))

            # checkpoint:
            if current_iteration % checkpoint_frequency == 0:
                print("Current Training Accuracy at Iteration {}: {}".format(current_iteration, train_acc[-1]))

                model_path = Directories.benchmark_models + str(name) + '_' + str(current_iteration) + '_' + str(batch_size) + \
                             '_' + str(learning_rate)
                torch.save(model.state_dict(), model_path)

            current_iteration += 1

    # plotting
    plt.title("Training Loss")
    plt.plot(iterations, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.savefig("results/training_loss.png")
    plt.close()

    plt.title("Training Accuracy")
    # Raw plot:
    # plt.plot(iterations, train_acc, label="Train")
    # savgol filter:
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_acc), polyorder=5, window_length=31), label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig("results/training_accuracy.png")
    plt.close()

    if validation_data is not None:
        plt.title("Validation Accuracy")
        plt.plot(validation_acc, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='best')
        plt.savefig("results/validation_accuracy.png")
        plt.close()

        plt.title("Validation Loss")
        plt.plot(validation_loss, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Loss")
        plt.legend(loc='best')
        plt.savefig("results/validation_loss.png")
        plt.close()
    print("Final Training Accuracy: {}".format(train_acc[-1]))