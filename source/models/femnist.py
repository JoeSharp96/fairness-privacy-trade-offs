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

class Net(nn.Module):

    def __init__(self):
        super().__init__()
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

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5), (0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, min_partition_size: int, distribution: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = get_partitioner(
            distribution,
            num_partitions,
            alpha
        )
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
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
    test_dataset = load_dataset("flwrlabs/femnist", split="train")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=64)

def ditto_train(net, lr, lmbda, global_params):
    """Bound the personalised model updates to not drift too far from the global model."""
    with torch.no_grad():
        for p, g_p, in zip(net.parameters(), global_params):
            update = p - lr * (p.grad + lmbda * torch.dist(p, g_p, p=2))
            p.copy_(update)
    return

def train(net, trainloader, epochs, lr, device, train_config, global_params = None):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running_loss = 0.0
    i = 0

    # DP Enabled training
    if train_config["dp"] and train_config["dp_mode"] == "local":
        privacy_engine = PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs = epochs,
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
        net.train()
        with BatchMemoryManager(
            data_loader=trainloader, 
            max_physical_batch_size=train_config["dp_max_physical_batch_size"], 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for _ in range(epochs):
                for batch in memory_safe_data_loader:
                    optimizer.zero_grad()
                    images = batch['image'].to(device)
                    labels = batch['character'].to(device)
                    loss = criterion(net(images), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if train_config["dp"] and (i+1) % 200 == 0:
                        epsilon = privacy_engine.get_epsilon(train_config["dp_delta"])
                    i += 1
                avg_trainloss = running_loss / (epochs * len(trainloader))
    # DP Disabled Training
    else:
        net.train()
        for _ in range(epochs):
            for batch in trainloader:
                optimizer.zero_grad()
                images = batch['image'].to(device)
                labels = batch['character'].to(device)
                loss = criterion(net(images), labels)
                loss.backward()

                # Ditto
                if global_params is not None:
                    ditto_train(net, lr, train_config["ditto_lambda"], global_params)

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
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


