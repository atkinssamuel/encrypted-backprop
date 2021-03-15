import torch
from src.benchmark.helpers import get_accuracy
from src.constants import Directories


def test_model(model, model_name, data):
    state = torch.load(Directories.benchmark_models + model_name)
    model.load_state_dict(state)
    accuracy = get_accuracy(model, data)
    print("Accuracy using model \"{}\" = {}%".format(model_name, round(accuracy * 100, 2)))
    return accuracy