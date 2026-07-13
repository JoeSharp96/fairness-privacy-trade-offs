"""pytorchexample: A Flower / PyTorch app."""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.preprocessor import Divider
from source.utils.datasets import get_partitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
from source.models.net import Net
# Non-IID settings:
# FedAvg
# LR =          0.1
# Batch size =  16
# Epochs =      1

# Q-FedAvg
# LR =          ?
# Batch size =  ?
# Epochs =      ?

# Ditto
# LR =          ?
# Batch size =  ?
# Epochs =      ?

# IID settings:
# FedAvg
# LR =          ?
# Batch size =  ?
# Epochs =      ?

# Q-FedAvg
# LR =          ?
# Batch size =  ?
# Epochs =      ?

# Ditto
# LR =          ?
# Batch size =  ?
# Epochs =      ?


class Femnist(Net):

    def __init__(self, lr, epochs, batch_size, num_partitions, distribution, alpha, partition_by = "Race", ditto = False, input_dim: int = 14):
        super(Femnist, self).__init__(lr, epochs, batch_size, num_partitions, distribution, alpha, partition_by, ditto)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128,62)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def test(self, testloader, device):
        """Validate the model on the test set."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["character"].to(device)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5), (0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, distribution: str):
    """Load partition FEMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        print("Fds empty")
        partitioner = get_partitioner(distribution=distribution, num_partitions=num_partitions, alpha=alpha, partition_by="writer_id", femnist=True)
        preprocessor = Divider(
            divide_config={"train": {"train":0.8, "test":0.2}}
        )
        # Other examples online use NaturalPartitioner. Might be worth looking into
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": partitioner},
            preprocessor=preprocessor,
            seed=42
        )
    partition = fds.load_partition(partition_id, split="train")
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset(distribution=None, batch_size = 16):
    """Load and split the centralized dataset"""
    global fds
    if fds is None:
        partitioner = get_partitioner(distribution, None, None, "writer_id", True)
        preprocessor = Divider(
            divide_config={"train": {"train":0.8, "test":0.2}}
        )
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train":partitioner},
            preprocessor=preprocessor,
            seed=42
        )
    central_test_data = fds.load_split("test")
    central_test_data = central_test_data.with_transform(apply_transforms)
    testloader = DataLoader(central_test_data, batch_size=batch_size)
    return testloader


