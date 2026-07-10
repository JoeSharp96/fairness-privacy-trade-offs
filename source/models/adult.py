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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
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


class Net(nn.Module):

    def __init__(self, input_dim: int = 14):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

fds = None  # Cache FederatedDataset

# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, min_partition_size: int, distribution: str):
    """Load partition Adult data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = get_partitioner(distribution=distribution, num_partitions=num_partitions, alpha=alpha, partition_by="race")
        preprocessor = Divider(
            divide_config={"train": {"train":0.8, "test":0.2}}
        )        
        # Other examples online use NaturalPartitioner. Might be worth looking into
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train": partitioner},
            preprocessor=preprocessor,
            seed=42
        )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    X = dataset.drop("income", axis=1)
    y = dataset["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_centralized_dataset(distribution=None):
    """Load and split the centralized dataset"""
    global fds
    if fds is None:
        partitioner = get_partitioner(distribution, 1, 1, 'race')
        preprocessor = Divider(
            divide_config={"train": {"train":0.8, "test":0.2}}
        )
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train":partitioner},
            preprocessor=preprocessor,
            seed=42
        )
    dataset = fds.load_split("test").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    X = dataset.drop("income", axis=1)
    y = dataset["income"]

    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    X = preprocessor.fit_transform(X)

    X_test_tensor = torch.tensor(X, dtype=torch.float32)
    y_test_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return test_loader
    

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
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
                for X_batch, y_batch in memory_safe_data_loader:
                    optimizer.zero_grad()
                    outputs = net(X_batch)
                    loss = criterion(outputs, y_batch)
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
            for X_batch, y_batch in trainloader:
                optimizer.zero_grad()
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
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
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            outputs = net(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy


