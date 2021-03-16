from src.benchmark.model import FCBenchmark
from src.constants import Files, Benchmark, Directories
from src.benchmark.train import train_model
from src.benchmark.test import test_model

import pandas as pd
import numpy as np
import glob


def run_benchmark():
    df = pd.read_pickle(Files.balanced_data)

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_model(FCBenchmark(), "FCBenchmark", training_data=train, validation_data=validate,
                batch_size=Benchmark.batch_size, epoch_count=Benchmark.epoch_count,
                shuffle=Benchmark.shuffle_flag, learning_rate=Benchmark.learning_rate,
                checkpoint_frequency=Benchmark.checkpoint_frequency)
    best_model = glob.glob(Directories.benchmark_best_model + "*")[0]
    test_model(FCBenchmark(), best_model, test)
