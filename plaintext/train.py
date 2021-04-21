from src.constants import Directories, Files
from src.helpers import round_sig, compute_accuracy
from os import path

import math
import os
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, name, training_data, validation_data=None, batch_size=1, epoch_count=1, learning_rate=0.01,
                checkpoint_frequency=5):
    # creating results folder if it doesn't exist
    if not path.exists(Directories.plaintext_results):
        os.mkdir(Directories.plaintext_results)
    num_batches = math.ceil(training_data.__len__() / batch_size)

    losses = []
    training_loss_averages = []
    validation_accuracies = []
    current_iteration = 0

    for epoch in range(epoch_count):
        batch_number = 0
        for batch in np.array_split(training_data, num_batches):
            inputs = batch.drop("Class", axis=1).values
            labels = batch["Class"].values
            for i in range(batch.shape[0]):
                model_input = inputs[i, :].reshape(1, -1)
                label = labels[i]
                output = model.single_sample_forward(model_input)

                losses.append(model.compute_loss(output, label))
            model.update_params()

            # checkpoint:
            if current_iteration % checkpoint_frequency == 0:
                # validation accuracy
                valid_inputs = validation_data.drop("Class", axis=1).values
                valid_labels = validation_data["Class"].values
                valid_predictions = model.batch_forward(valid_inputs)
                validation_accuracies.append(round(100*compute_accuracy(valid_predictions, valid_labels), 2))

                # progress (0 -> 1)
                progress = epoch / epoch_count + (batch_number / num_batches) / epoch_count
                training_loss_average = np.average(losses[-checkpoint_frequency:])
                if not math.isnan(training_loss_average):
                    training_loss_average = round_sig(training_loss_average, 3)
                training_loss_averages.append(training_loss_average)
                print(f"Progress: {round(progress * 100, 2)}%, "
                      f"Batch {batch_number}/{num_batches}, "
                      f"Epoch {epoch}/{epoch_count}, "
                      f"Training Loss = {training_loss_average}, "
                      f"Validation Accuracy = {validation_accuracies[-1]}%")

            batch_number += 1
            current_iteration += 1

    # training loss plotting
    plt.plot(training_loss_averages)
    plt.title("Training Loss")
    plt.xlabel("Checkpoint Iteration")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.savefig(Files.plaintext_training_loss)
    plt.close()

    # validation accuracy plotting
    plt.plot(validation_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Checkpoint Iteration")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.savefig(Files.plaintext_validation_accuracy)
    plt.close()

    print("Hello World")