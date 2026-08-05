
import torch
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
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
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

    def train(self, trainloader: DataLoader, device: torch.device) -> float:
        running_loss = 0.0
        self.model.train()
        for _ in range(self.model.epochs):
            for X_batch, y_batch in trainloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                self.model.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.model.criterion(outputs, y_batch)
                loss.backward()
                self.model.optimizer.step()
                running_loss += loss.item()
        return running_loss

    def fit(self, device: torch.device, train_config: RecordDict) -> float:
        """Set criterion and optimizer. If DP enabled, set metrics. Then run the training method."""
        self.model.to(device)  # move model to GPU if available
        self.model.criterion = torch.nn.BCELoss().to(device)
        self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        if train_config["dp"]:
            privacy_engine = PrivacyEngine()
            self.model, self.model.optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.model.optimizer,
                data_loader=self.trainloader,
                epochs = self.model.epochs,
                target_epsilon=train_config["dp_epsilon"],
                target_delta=train_config["dp_delta"],
                max_grad_norm=train_config["dp_max_grad_norm"],
            )
            with BatchMemoryManager(
                data_loader=self.trainloader,
                max_physical_batch_size=train_config["dp_max_physical_batch_size"],
                optimizer=self.model.optimizer
            ) as memory_safe_data_loader:
                running_loss = self.train(memory_safe_data_loader, device)
        else:
            running_loss = self.train(self.trainloader, device)
        avg_trainloss = running_loss / (self.model.epochs * len(self.trainloader))
        return avg_trainloss

    def test(self, device: torch.device)->tuple[float, float, float, float, float, float] | tuple[float, float]:
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

def skew_and_split(sensitive_feature: str, sensitive_value: str | float, _skew: float = 0.3, seed: int = 42, global_train: bool = False):
    """Preprocessor for federated dataset"""
    def skew(dataset_dict: DatasetDict)->DatasetDict:
        # Check if Huggingface dataset already has a "test" split
        if "test" not in dataset_dict:
            split = Divider(
                divide_config={"train":0.8,"test":0.2},
                divide_split="train"
            )
            dataset_dict = split(dataset_dict)
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
            dataset=dataset_url,
            partitioners={"train": partitioner},
            preprocessor=preprocessor,
            seed=seed
        )
        # need to add an if statement here to only save the partition if I'm saving the model...
        save_partitions(sensitive_feature, sensitive_value, fds, seed, output_directory, num_partitions, skew, alpha)

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset.dropna(inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    drop_columns.append(target_feature)
    X = dataset.drop(drop_columns, axis=1)
    y = dataset[target_feature]
    sensitive_col_index = X.columns.get_loc(sensitive_feature)

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

    return train_loader, test_loader, sensitive_col_index
