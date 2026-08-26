
import torch
import gc
import math
import pandas as pd
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
from source.models.adult import Adult
from source.models.compas import Compas
from source.utils.reporting import save_partitions
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.preprocessor import Divider
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split
from flwr.app import RecordDict
from datasets import DatasetDict
from torch.utils.data import DataLoader, TensorDataset

class Client:
    """Creates a client class to train a local model. Local model is determined by the dataset used."""
    def __init__(
            self,
            partition_id: int,
            seed: int,
            lr: float, 
            epochs: int, 
            batch_size: int, 
            num_partitions: int, 
            distribution: str, 
            alpha: float, 
            sensitive_feature: str,
            sensitive_value: str,
            skew: float,
            target_feature: str,
            dataset: str
            ):
        self.partition_id = partition_id
        self.seed = seed
        self.target_feature = target_feature
        self.dataset = dataset
        self.testloader = None
        self.trainloader = None
        self.sensitive_col_index = None
        self.drop_columns = []
        self.model = self.load_model(lr, epochs, batch_size, num_partitions, distribution, alpha, sensitive_feature, sensitive_value, skew)

    def load_model(
            self,
            lr: int, 
            epochs: int, 
            batch_size: int, 
            num_partitions: int, 
            distribution: str, 
            alpha: float, 
            sensitive_feature: str, 
            sensitive_value: str, 
            skew: float
            )->Adult | Compas:
        """Loads a custom ML model to the client."""
        if self.dataset == "adult":
            self.partition_key = "train"
            self.dataset_url = "scikit-learn/adult-census-income"
            self.preprocessor = skew_and_split(
                sensitive_feature=sensitive_feature,
                sensitive_value=sensitive_value
                )
            return Adult(lr, epochs, batch_size, num_partitions, distribution, alpha, sensitive_feature, sensitive_value, skew)
        else:
            self.drop_columns = ["age","race:African-American", "race:Asian", "race:Hispanic", "race:Native_American", "race:Other"]
            self.partition_key = "test"
            self.dataset_url = "imodels/compas-recidivism"
            self.preprocessor = None
            return Compas(lr, epochs, batch_size, num_partitions, distribution, alpha, sensitive_feature, sensitive_value, skew)

    def load_data(self, output_dir: str) -> None:
        """Calls a load data function. Clients are stateless and would require loading the FederatedDatateset each round. load_data() stores the loaded dataset that can be accessed by all clients."""
        self.trainloader, self.testloader, self.sensitive_col_index = load_data(
            partition_id=self.partition_id, 
            num_partitions=self.model.num_partitions, 
            batch_size=self.model.batch_size, 
            alpha=self.model.alpha, 
            sensitive_feature=self.model.sensitive_feature,
            sensitive_value=self.model.sensitive_value,
            skew=self.model.skew,
            seed=self.seed,
            target_feature=self.target_feature,
            output_directory=output_dir,
            dataset_url = self.dataset_url,
            drop_columns = self.drop_columns
            )

    def train(self, trainloader: DataLoader, device: torch.device, model, epochs) -> float:
        """Trains tabular data for n epochs. Returns running loss"""
        running_loss = 0.0
        model.train()
        for _ in range(epochs):
            for X_batch, y_batch in trainloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                model.optimizer.zero_grad()
                outputs = model(X_batch)
                loss = model.criterion(outputs, y_batch)
                loss.backward()
                model.optimizer.step()
                running_loss += loss.item()
        return running_loss

    def fit(self, device: torch.device, train_config: RecordDict) -> float:
        """Set criterion and optimizer. If DP enabled, set epislon and clipping threshold. Then run the training method."""
        self.model.to(device)  # move model to GPU if available
        self.model.criterion = torch.nn.BCELoss().to(device)
        self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        if train_config["dp"]:
            privacy_engine = PrivacyEngine()
            model, model.optimizer, trainloader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.model.optimizer,
                data_loader=self.trainloader,
                epochs = self.model.epochs,
                target_epsilon=train_config["dp_epsilon"],
                target_delta=train_config["dp_delta"],
                max_grad_norm=train_config["dp_max_grad_norm"],
            )
            with BatchMemoryManager(
                data_loader=trainloader,
                max_physical_batch_size=train_config["dp_max_physical_batch_size"],
                optimizer=model.optimizer
            ) as memory_safe_data_loader:
                running_loss = self.train(memory_safe_data_loader, device, model, self.model.epochs)
        else:
            running_loss = self.train(self.trainloader, device, self.model, self.model.epochs)
        avg_trainloss = running_loss / (self.model.epochs * len(self.trainloader))
        return avg_trainloss

    def test(self, device: torch.device)->tuple[float, ...]:
        """Validate the model on the test set."""
        self.model.to(device)
        criterion = torch.nn.BCELoss()
        correct, loss = 0, 0.0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = self.model(X_batch)
                batch_loss = criterion(outputs, y_batch)
                loss += batch_loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        loss = loss / len(self.testloader)
        return loss, accuracy

fds = None  # Cache FederatedDataset

def clear_fds()->None:
    """To ensure there is no memory leakage between, fds is cleared at the end of each run by the server."""
    global fds
    del fds
    gc.collect()

def skew_and_split(sensitive_feature: str, sensitive_value: str | float, _skew: float = 0.3, seed: int = 42, global_train: bool = False):
    """Higher-order function to preprocess federated dataset. Returns the skew function"""
    def skew(dataset_dict: DatasetDict)->DatasetDict:
        """Minority group is reduced to match the _skew value. Updated dataset_dict is returned to the client."""
        # Check if Huggingface dataset already has a "test" split
        if "test" not in dataset_dict:
            split = Divider(
                divide_config={"train":0.8,"test":0.2},
                divide_split="train"
            )
            dataset_dict = split(dataset_dict)
        # Convert HuggingFace dataset to pandas for preprocessing
        dataset = dataset_dict["train"]
        # Split dataset into minority and majority group
        minority_dataset = dataset.filter(lambda x: x[sensitive_feature] == sensitive_value)
        majority_dataset = dataset.filter(lambda x: x[sensitive_feature] != sensitive_value)
        # Reduce total minority skew.
        total_majority = len(majority_dataset)
        required_minority = math.floor((total_majority * _skew) / (1 - _skew))
        minority_dataset = minority_dataset.shuffle(seed=seed).select(range(required_minority))
        # Return skewed dataset.
        skewed_dataset = concatenate_datasets([majority_dataset, minority_dataset]).shuffle(seed=seed)
        dataset_dict["train"] = skewed_dataset
        dataset_dict.pop("test")
        return dataset_dict
    return skew

# Add the partition_by variable. To run this on two datasets it will have to be variable.
def load_data(
        partition_id: int, 
        num_partitions: int, 
        batch_size: int, 
        alpha: float, 
        sensitive_feature: str, 
        sensitive_value: str, 
        skew: int, 
        seed: int, 
        output_directory: str, 
        target_feature: str,
        dataset_url: str,
        drop_columns: list,
        )->tuple[DataLoader,DataLoader]:
    """Load partition Adult data."""
    # Only initialize `FederatedDataset` once
    global fds
    # Check if fds has already been initialised. If not, load the federated dataset from huggingface
    if fds is None:
        # Set dirichlet partititioner. Alpha value determines how uniform distribution across clients is. alpha=0.2, extreme non-iid. alpha=500, iid.
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=sensitive_feature,
            alpha=alpha,
            seed=seed
        )
        # skew data by sensitive feature to a set ratio (skew). 
        preprocessor = skew_and_split(sensitive_feature, sensitive_value, skew, seed)    
        fds = FederatedDataset(
            dataset=dataset_url,
            partitioners={"train": partitioner},
            preprocessor=preprocessor,
            seed=seed
        )
        # Save a sample of the data for each partition.
        save_partitions(sensitive_feature, sensitive_value, fds, seed, output_directory, num_partitions, skew, alpha)
    # Load the local dataset for the client.
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    # Convert categorical columns to ordinal for transformation to tensor.
    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    # Split dataset in features (X) and target (y)
    drop_columns.append(target_feature)
    X = dataset.drop(drop_columns, axis=1)
    y = dataset[target_feature]
    # Get the sensitive column index number for evaluating fairness.
    sensitive_col_index = X.columns.get_loc(sensitive_feature)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Standardize numeric values using StandardScaler()
    numeric_features = X.select_dtypes(include=["float64", "int64", "bool"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert data into tensor format to be used in NN.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Initalise a test and training TensorDataset for local data.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, sensitive_col_index
