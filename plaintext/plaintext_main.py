from src.constants import Files, Plaintext
from plaintext.train import train_model
from plaintext.models import MultiLayerBenchmark, SingleLayerBenchmark

import pandas as pd
import numpy as np


def run_plaintext():
    """
    Trains and tests a manual implementation of a fully-connected neural network
    :return: None
    """
    df = pd.read_pickle(Files.balanced_data)

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_model(MultiLayerBenchmark(D=30, H=60, lr=Plaintext.learning_rate), "MultiLayerBenchmark", training_data=train,
                validation_data=validate, batch_size=Plaintext.batch_size, epoch_count=Plaintext.epoch_count,
                learning_rate=Plaintext.learning_rate, checkpoint_frequency=Plaintext.checkpoint_frequency)
