from src.benchmark.helpers import get_accuracy, get_loss, round_sig
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


def train_model(model, name, training_data, validation_data=None, batch_size=1, epoch_count=1, shuffle=False,
                learning_rate=0.01, checkpoint_frequency=5):
    # creating results folder if it doesn't exist
    if not path.exists(Directories.benchmark_results):
        os.mkdir(Directories.benchmark_results)

    # clearing models folder
    if path.exists(Directories.benchmark_models):
        shutil.rmtree(Directories.benchmark_models)

    os.mkdir(Directories.benchmark_models)
    os.mkdir(Directories.benchmark_other_models)
    os.mkdir(Directories.benchmark_best_model)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iterations, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    min_valid_loss = None
    best_model = None
    best_model_string = None

    # training
    current_iteration = 0  # the number of iterations

    num_batches = math.ceil(training_data.__len__() / batch_size)

    for epoch in range(epoch_count):
        batch_number = 0
        for batch in np.array_split(training_data, num_batches):
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

                # progress (0 -> 1)
                progress = epoch/epoch_count + (batch_number/num_batches)/epoch_count
                print(f"Progress: {round(progress*100, 2)}%, "
                      f"Batch {batch_number}/{num_batches}, "
                      f"Epoch {epoch}/{epoch_count}, "
                      f"Training Accuracy = {round(100 * train_acc[-1], 2)}%, "
                      f"Training Loss = {round_sig(losses[-1], 3)}, "
                      f"Validation Accuracy = {round(100 * validation_acc[-1] , 2)}%, "
                      f"Validation Loss = {round_sig(validation_loss[-1], 3)}")

                model_string = str(name) + '_' + str(current_iteration) + '_' + str(batch_size) + '_' + \
                               str(learning_rate)

                torch.save(model.state_dict(), Directories.benchmark_other_models + model_string)

                if min_valid_loss is None or validation_loss[-1] < min_valid_loss:
                    min_valid_loss = validation_loss[-1]
                    best_model = model
                    best_model_string = model_string

            current_iteration += 1
            batch_number += 1


    # plotting
    plt.title("Training Loss")
    plt.plot(iterations, losses, label="Train")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.savefig(Files.benchmark_training_loss)
    plt.close()

    plt.title("Training Accuracy")
    # Raw plot:
    # plt.plot(iterations, train_acc, label="Train")
    # savgol filter:
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_acc), polyorder=5, window_length=31), label="Train")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig(Files.benchmark_training_accuracy)
    plt.close()

    if validation_data is not None:
        plt.title("Validation Accuracy")
        plt.plot(validation_acc, label="Validation")
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='best')
        plt.savefig(Files.benchmark_validation_accuracy)
        plt.close()

        plt.title("Validation Loss")
        plt.plot(validation_loss, label="Validation")
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("Validation Loss")
        plt.legend(loc='best')
        plt.savefig(Files.benchmark_validation_loss)
        plt.close()


    torch.save(best_model.state_dict(), Directories.benchmark_best_model + best_model_string)

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print(f"Best Model: {best_model_string}")
