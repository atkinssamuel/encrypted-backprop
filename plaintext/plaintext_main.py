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
    def string_to_int(entry):
        if entry == "positive":
            return 1
        return 0

    df = pd.read_csv(Files.androgen_data, sep=";", header=None)
    df.rename(columns={1024: "Class"}, inplace=True)
    df["Class"] = df["Class"].map(string_to_int)

    num_features = df.shape[1]-1

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_model(MultiLayerBenchmark(D=num_features, H=num_features*2, lr=Plaintext.learning_rate),
                "MultiLayerBenchmark", training_data=train,
                validation_data=validate, batch_size=Plaintext.batch_size, epoch_count=Plaintext.epoch_count,
                learning_rate=Plaintext.learning_rate, checkpoint_frequency=Plaintext.checkpoint_frequency)
