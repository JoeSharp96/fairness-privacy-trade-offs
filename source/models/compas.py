"""pytorchexample: A Flower / PyTorch app."""
import random
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.preprocessor import Divider
from source.utils.datasets import get_partitioner
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from source.models.net import Net
from datasets import concatenate_datasets
from sklearn.metrics import accuracy_score, confusion_matrix
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
            ditto = False, 
            input_dim: int = 14):
        super(Compas, self).__init__(
            lr=lr, 
            epochs=epochs, 
            batch_size=batch_size, 
            num_partitions=num_partitions, 
            distribution=distribution, 
            alpha=alpha, 
            sensitive_feature=sensitive_feature, 
            sensitive_value=sensitive_value, 
            skew=skew, 
            ditto=ditto
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

    def group_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, x_sensitive_feature: np.ndarray):
        # Gets the unique groups. As the data was originally a tensor, it works out the counts to determine the minority and majority groups.
        unique_groups, counts = np.unique(x_sensitive_feature, return_counts=True)
        if counts[0] > counts[1]:
            majority_group = unique_groups[0]
            minority_group = unique_groups[1]
        else:
            minority_group = unique_groups[0]
            majority_group = unique_groups[1]
        min_condition = (x_sensitive_feature == minority_group)
        maj_condition = (x_sensitive_feature == majority_group)
        x_sensitive_feature[min_condition] = 0.0
        x_sensitive_feature[maj_condition] = 1.0
        unique_groups, counts = np.unique(x_sensitive_feature, return_counts=True)
        metrics = {}
        for value in unique_groups:
            mask = (x_sensitive_feature == value)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            # Demographic Parity
            ppr = np.mean(y_pred_group == 1)

            if np.sum(y_true_group == 1) > 0:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0.,1.]).ravel()
                tpr = tp / (tp + fn)
            else:
                tpr = np.nan

            acc = accuracy_score(y_true_group, y_pred_group)

            metrics[value] = {'PPR': ppr, 'TPR': tpr, 'ACC': acc}

        group_ids = list(metrics.keys())
        print(group_ids)
        g1, g2 = group_ids[0], group_ids[1]
        print(f'=== Group Metrics ===')
        for key in metrics.keys():
            print(f"Group {key}: PPR = {metrics[key]['PPR']:.3f}, TPR = {metrics[key]['TPR']:.3f}, ACC = {metrics[key]['ACC']:.3f}")
        print("=== Fairness Violations ===")
        print(f"Demographic Parity: {abs(metrics[g1]['PPR']-metrics[g2]['PPR']):.3f}")
        print(f"Equalised Odds: {abs(metrics[g1]['TPR']-metrics[g2]['TPR']):.3f}")
        print(f"Equalised Accuracy: {abs(metrics[g1]['ACC']-metrics[g2]['ACC']):.3f}")
        demographic_parity = float(abs(metrics[g1]['PPR']-metrics[g2]['PPR']))
        equalised_odds = float(abs(metrics[g1]['TPR']-metrics[g2]['TPR']))
        minority_acc = float(metrics[0.]['ACC'])
        majority_acc = float(metrics[1.]['ACC'])
        return demographic_parity, equalised_odds, minority_acc, majority_acc
        

    
    def test(self, testloader, device, globaltest=False):
        """Validate the model on the test set."""
        self.to(device)
        criterion = torch.nn.BCELoss()
        correct, loss = 0, 0.0
        total = 0
        y_true = []
        y_pred = []
        x_sensitive_feature = []
        with torch.no_grad():
            # y_batch is true
            # prediction is prediction
            # I can collect the fairness metrics by examining these in relationship to the sensitive attributes
            # Maybe collect all results into a list?
            for X_batch, y_batch in testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = self(X_batch)
                batch_loss = criterion(outputs, y_batch)
                loss += batch_loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                y_true.append(y_batch.flatten().float())
                y_pred.append(predicted.flatten().float())
                x_sensitive_feature.append(X_batch[:, 8].flatten())
        accuracy = correct / total
        loss = loss / len(testloader)
        if globaltest:
            y_true = torch.cat(y_true, dim=0).cpu()
            y_pred = torch.cat(y_pred, dim=0).cpu()
            x_sensitive_feature = torch.cat(x_sensitive_feature, dim=0).cpu()
            dp, eo, min_acc, maj_acc = self.group_fairness_metrics(y_true.numpy(), y_pred.numpy(), x_sensitive_feature.numpy())
            return loss, accuracy, dp, eo, min_acc, maj_acc
        return loss, accuracy

fds = None  # Cache FederatedDataset

def skew_and_split(sensitive_feature: str, sensitive_value: str|float, _skew: float = 0.3, seed: int = 42, global_train: bool = False):

    def skew(dataset_dict):
        # Convert HuggingFace dataset to pandas for preprocessing
        dataset = dataset_dict["train"]
        minority_dataset = dataset.filter(lambda x: x[sensitive_feature] == sensitive_value)
        majority_dataset = dataset.filter(lambda x: x[sensitive_feature] != sensitive_value)
        total_majority = len(majority_dataset)
        required_minority = math.floor((total_majority * _skew) / (1 - _skew))
        minority_dataset = minority_dataset.shuffle(seed=seed).select(range(required_minority))
        total_minority = len(minority_dataset)
        print(f"""Total training data: {total_majority + total_minority}
        Minority = {total_minority} | {(total_minority / (total_minority+total_majority))*100}%
        Majority = {total_majority} | {(total_majority / (total_minority+total_majority))*100}%""")
        
        skewed_dataset = concatenate_datasets([majority_dataset, minority_dataset]).shuffle(seed=seed)
        dataset_dict["train"] = skewed_dataset
        return dataset_dict
    return skew

def save_partitions(sensitive_feature: str, sensitive_value: str, fds: FederatedDataset, seed: int, output_directory: str, num_partitions: int):
    df = pd.read_csv(f"output/{output_directory}/{seed}/data_sample.csv")
    if len(df.index) < num_partitions:
        minority_count = []
        majority_count = []
        for i in range(num_partitions):
            dataset = fds.load_partition(i, "train").with_format("pandas")[:]
            minority_count.append(dataset.where(dataset[sensitive_feature]==sensitive_value).count()[sensitive_feature])
            majority_count.append(dataset.where(dataset[sensitive_feature]!=sensitive_value).count()[sensitive_feature])
        df2 = pd.DataFrame(data={"min":minority_count, "maj":majority_count})
        df2.to_csv(f"output/{output_directory}/{seed}/data_sample.csv", mode="w", header=True, index=False)



# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, sensitive_feature: str, sensitive_value: str, skew: int, seed: int, output_directory: str):
    """Load partition compas data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=sensitive_feature,
            alpha=alpha,
            seed=seed
        )
        preprocessor = skew_and_split(sensitive_feature, sensitive_value, skew, seed)    
        # Other examples online use NaturalPartitioner. Might be worth looking into
        fds = FederatedDataset(
            dataset="imodels/compas-recidivism",
            partitioners={"train": partitioner},
            preprocessor=preprocessor,
            seed=seed
        )
        save_partitions(sensitive_feature, sensitive_value, fds, seed, output_directory, num_partitions)

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    X = dataset.drop(["is_recid","age","race:African-American", "race:Asian", "race:Hispanic", "race:Native_American", "race:Other"], axis=1)
    y = dataset["is_recid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    numeric_features = X.select_dtypes(include=["float64", "int64", "bool"]).columns
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

def load_centralized_dataset(num_partitions: int, batch_size: int, alpha: float, sensitive_feature: str, sensitive_value: str, skew: float, seed: int):
    """Load and split the centralized dataset"""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=sensitive_feature,
            alpha=alpha,
            seed=seed
        )
        fds = FederatedDataset(
            dataset="imodels/compas-recidivism",
            partitioners={"test": partitioner},
            seed=seed
        )
    dataset = fds.load_split("test").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    X = dataset.drop(["is_recid","age","race:African-American", "race:Asian", "race:Hispanic", "race:Native_American", "race:Other"], axis=1)
    y = dataset["is_recid"]

    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    X = preprocessor.fit_transform(X)

    X_test_tensor = torch.tensor(X, dtype=torch.float32)
    y_test_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return test_loader
