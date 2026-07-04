"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr.serverapp.strategy import Result
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
from flwr.app import UserConfig
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


FM_NORMALIZATION = ((0.1307,), (0.3081,))
TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])

# I probably will need a unique model to work with every dataset I'm going to use :)
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [TRANSFORMS(img) for img in batch["image"]]
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
            seed=42
        )
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
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
    test_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=64)

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

def ditto_train(net, lr, lmbda, global_params):
    with torch.no_grad():
        for p, g_p, in zip(net.parameters(), global_params):
            update = p - lr * (p.grad + lmbda * torch.dist(p, g_p, p=2))
            p.copy_(update)
    return


def train(net, trainloader, epochs, lr, device, max_physical_batch_size, context, ditto = None, global_params = None):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
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
                    images = batch['image'].to(device)
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
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                loss = criterion(net(images), labels)
                loss.backward()

                # Ditto
                if ditto is not None:
                    ditto_train(net, ditto["lr"], ditto["lambda"], global_params)


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
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() # For ditto, I'd need to add the lambda value by the model distance norm value
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def output_dir(config: UserConfig) -> tuple[Path, str]:
    """Create directory for output graph and data"""
    current_time = datetime.now()
    out_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    path = Path.cwd() / f"output/{out_dir}"
    path.mkdir(parents=True, exist_ok=False)

    with open(f"{path}/run_config.json","w",encoding="utf-8") as fp:
        json.dump(config,fp)

    return path

def save_metrics(result: Result, save_path, rounds):
    """Save metrics"""
    results = {}
    for i in range(1,rounds+1):
        train_metrics = dict(result.train_metrics_clientapp.get(i,{}))
        eval_client_metrics = dict(result.evaluate_metrics_clientapp.get(i,{}))
        eval_server_metrics = dict(result.evaluate_metrics_serverapp.get(i,{}))
        round_result = {
            "train_loss": train_metrics["train_loss"],
            "eval_client_loss": eval_client_metrics["eval_loss"],
            "eval_client_acc": eval_client_metrics["eval_acc"],
            "eval_server_loss": eval_server_metrics["loss"],
            "eval_server_acc": eval_server_metrics["accuracy"]
        }
        results[i] = round_result
    
    with open(f"{save_path}/results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp)

def save_graphs(save_path, rounds):
    """Creates matplotlib graphs of results and saves them as JPG files"""
    with open(f"{save_path}/results.json", "r") as jsonfile:
        df = pd.read_json(jsonfile, orient="index")
        #results = json.load(jsonfile)
    
    with open(f"{save_path}/run_config.json","r") as jsonfile:
        config = json.load(jsonfile)

    epochs = config['local-epochs']
    if config['dp-enabled']:
        epsilon = config['epsilon']
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nε = {epsilon}"
    else:
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nNon-DP"

    plt.figure(figsize=(5, 5))
    plt.plot(df.index, df['eval_client_acc'], marker='o', color='b', label='Aggregate Client Accuracy')
    plt.plot(df.index, df['eval_server_acc'], marker='x', color='r', label='Global Accuracy')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_path}/eval_acc.jpg")
