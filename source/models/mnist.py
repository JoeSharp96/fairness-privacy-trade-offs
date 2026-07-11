"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
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

class Mnist(Net):

    def __init__(self, lr, epochs, batch_size, num_partitions, distribution, alpha, partition_by = "label", ditto = False):
        super().__init__(lr, epochs, batch_size, num_partitions, distribution, alpha, partition_by = "label", ditto = False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def fit(self, trainloader, device, train_config):
        """Train the model on the training set."""
        self.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        running_loss = 0.0
        i = 0

        # DP Enabled training
        if train_config["dp"] and train_config["dp_mode"] == "local":
            privacy_engine = PrivacyEngine()
            self, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
                module=self.net,
                optimizer=optimizer,
                data_loader=trainloader,
                epochs = self.epochs,
                clipping=train_config["dp_clipping"],
                target_epsilon=train_config["dp_epsilon"],
                target_delta=train_config["dp_delta"],
                max_grad_norm=train_config["dp_max_grad_norm"],
                target_unclipped_quantile = 0.5,
                clipbound_learning_rate = 0.2,
                max_clipbound = 100.0,
                min_clipbound = train_config["dp_min_bound"],
                unclipped_num_std = 2.0
            )
            self.train()
            with BatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=train_config["dp_max_physical_batch_size"], 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for _ in range(self.epochs):
                    for batch in memory_safe_data_loader:
                        optimizer.zero_grad()
                        images = batch['image'].to(device)
                        labels = batch['label'].to(device)
                        loss = criterion(self(images), labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if train_config["dp"] and (i+1) % 200 == 0:
                            epsilon = privacy_engine.get_epsilon(train_config["dp_delta"])
                        i += 1
                    avg_trainloss = running_loss / (self.epochs * len(trainloader))
        # DP Disabled Training
        else:
            self.train()
            for _ in range(self.epochs):
                for batch in trainloader:
                    optimizer.zero_grad()
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    loss = criterion(self(images), labels)
                    loss.backward()

                    # Ditto
                    if self.global_params is not None:
                        self.ditto_train()

                    optimizer.step()
                    running_loss += loss.item()
                avg_trainloss = running_loss / (self.epochs * len(trainloader))
        return avg_trainloss
    
    def ditto_train(self):
        """Bound the personalised model updates to not drift too far from the global model."""
        with torch.no_grad():
            for p, g_p, in zip(self.parameters(), self.global_params):
                update = p - self.lr * (p.grad + self.lmbda * torch.dist(p, g_p, p=2))
                p.copy_(update)
        return
    
    def test(self, testloader, device):
        """Validate the model on the test set."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
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
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = get_partitioner(
            distribution,
            num_partitions,
            alpha,
            "label"
        )
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(distribution=None, batch_size=16):
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("ylecun/mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=batch_size)
