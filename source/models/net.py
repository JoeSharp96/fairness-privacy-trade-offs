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

class Net(nn.Module):
    def __init__(self, lr: float, epochs: int, batch_size: int, num_partitions: int, distribution: str, alpha: float, partition_by: str, ditto: bool = False):
        super(Net, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.distribution = distribution
        self.alpha = alpha
        self.partition_by = partition_by
        self.ditto = ditto
        self.lmbda = None
        self.global_params = None