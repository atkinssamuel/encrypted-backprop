from src.helpers import ReLU, sigmoid, xavier_init

import numpy as np

class PlaintextBenchmark:
    def __init__(self, D, H):
        initialization_scalar = np.sqrt(2/D)

        self.W1 = xavier_init(D, H, initialization_scalar)
        self.b1 = xavier_init(1, H, initialization_scalar)

        self.W2 = xavier_init(H, 1, initialization_scalar)
        self.b2 = xavier_init(1, 1, initialization_scalar)

        self.a1, self.z1, self.a2, self.z2 = [None] * 4

        self.loss = None

    def single_sample_forward(self, X):
        self.a1 = X @ self.W1 + self.b1
        self.z1 = ReLU(self.a1)

        self.a2 = self.z1 @ self.W2 + self.b2
        self.z2 = sigmoid(self.a2)

        return self.z2

    def single_sample_backward(self):
        # TODO
        self.loss = None
        return
