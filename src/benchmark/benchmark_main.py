from src.benchmark.model import FCBenchmark
from src.constants import Files
from src.benchmark.train import train_model
from src.benchmark.test import test_model
import pandas as pd


def run_benchmark():
    df = pd.read_pickle(Files.balanced_data)
    train_model(FCBenchmark(), "FCBenchmark", df, batch_size=256, epoch_count=2000, shuffle=True, learning_rate=0.001,
                checkpoint_frequency=20)
    model_name = "FCBenchmark_20_256_0.001"
    test_model(FCBenchmark(), model_name, df)
