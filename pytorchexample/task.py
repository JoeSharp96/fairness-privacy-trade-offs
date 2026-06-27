"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, min_partition_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by='label',
            alpha=alpha,
            min_partition_size=min_partition_size,
            self_balancing=True,
            seed=5
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
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


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

"""
def train(net, trainloader, epochs, lr, device, optimizer, max_physical_batch_size,epsilon,delta,privacy_engine):
    Train the model on the training set.
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running_loss = 0.0
    i = 0
    net.train()
    with BatchMemoryManager(
        data_loader=trainloader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for batch in memory_safe_data_loader:
            optimizer.zero_grad()
            images = batch['img'].to(device)
            labels = batch['label'].to(device)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(delta)
            i += 1
        avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss
"""

def train(net, trainloader, epochs, lr, device, max_physical_batch_size, context):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    running_loss = 0.0
    i = 0
    # DP Enabled training
    if context.run_config["dp-enabled"]:
        privacy_engine = PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs = epochs,
            target_epsilon=context.run_config["epsilon"],
            target_delta=context.run_config["delta"],
            max_grad_norm=context.run_config["max-grad-norm"]
        )
        net.train()
        with BatchMemoryManager(
            data_loader=trainloader, 
            max_physical_batch_size=max_physical_batch_size, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for _ in range(epochs):
                for batch in memory_safe_data_loader:
                    optimizer.zero_grad()
                    images = batch['img'].to(device)
                    labels = batch['label'].to(device)
                    loss = criterion(net(images), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if context.run_config["dp-enabled"] and (i+1) % 200 == 0:
                        epsilon = privacy_engine.get_epsilon(context.run_config["delta"])
                    i += 1
                avg_trainloss = running_loss / (epochs * len(trainloader))
    # DP Disabled Training
    else:
        net.train()
        for _ in range(epochs):
            for batch in trainloader:
                optimizer.zero_grad()
                images = batch['img'].to(device)
                labels = batch['label'].to(device)
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
