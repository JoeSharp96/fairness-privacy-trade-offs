"""pytorchexample: A Flower / PyTorch app."""
import torch.nn as nn
from source.models.net import Net

class Compas(Net):

    def __init__(
            self, 
            lr: float, 
            epochs: int, 
            batch_size: int, 
            num_partitions: int, 
            distribution: str, 
            alpha: float, 
            sensitive_feature: str, 
            sensitive_value: str,
            skew: float,
            input_dim: int = 14
            ):
        super(Compas, self).__init__(
            lr=lr, 
            epochs=epochs, 
            batch_size=batch_size, 
            num_partitions=num_partitions, 
            distribution=distribution, 
            alpha=alpha, 
            sensitive_feature=sensitive_feature, 
            sensitive_value=sensitive_value, 
            skew=skew
            )
        self.layer1 = nn.Linear(input_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x