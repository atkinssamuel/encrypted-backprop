import torch


class FCBenchmark(torch.nn.Module):
    def __init__(self):
        super(FCBenchmark, self).__init__()

        input_size = 30
        hidden_layer_size = input_size * 2
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class SingleLayerBenchmark(torch.nn.Module):
    def __init__(self):
        super(SingleLayerBenchmark, self).__init__()

        input_size = 30
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc1(x))
