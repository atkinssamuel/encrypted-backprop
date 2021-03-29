from benchmark.models import FCBenchmark, SingleLayerBenchmark
from benchmark.train import train_model
from benchmark.test import test_model
from src.constants import Files, Benchmark, Directories

import pandas as pd
import numpy as np
import glob


def run_benchmark():
    """
    Trains and tests a fully-connected benchmark model using PyTorch
    :return: None
    """
    df = pd.read_pickle(Files.balanced_data)

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_model(SingleLayerBenchmark(), "SingleLayerBenchmark", training_data=train, validation_data=validate,
                batch_size=Benchmark.batch_size, epoch_count=Benchmark.epoch_count,
                learning_rate=Benchmark.learning_rate, checkpoint_frequency=Benchmark.checkpoint_frequency)

    best_model_path = glob.glob(Directories.benchmark_best_model + "*")[0]
    test_model(SingleLayerBenchmark(), best_model_path, test)
