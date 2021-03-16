import torch
import torch.nn as nn
import numpy as np
import math


def round_sig(x, sig=2):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)


def get_accuracy(model, data):
    correct = 0
    total = 0
    inputs = torch.tensor(data.drop("Class", axis=1).values).reshape(shape=(-1, 1, 30))
    labels = torch.tensor(data["Class"].values).reshape(shape=(-1, 1, 1))
    outputs = model(inputs.float()).detach().numpy()
    outputs = np.round(outputs)
    labels = labels.detach().numpy()
    total += inputs.shape[0]
    wrong = np.sum(np.abs(outputs - labels))
    correct += inputs.shape[0] - wrong
    return correct/total


def get_loss(model, data):
    criterion = nn.BCELoss()
    inputs = torch.tensor(data.drop("Class", axis=1).values).reshape(shape=(-1, 1, 30))
    labels = torch.tensor(data["Class"].values).reshape(shape=(-1, 1, 1))
    outputs = model(inputs.float())
    loss = criterion(outputs, labels.float())
    return float(loss) / data.__len__()