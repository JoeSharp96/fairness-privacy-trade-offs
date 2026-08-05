import torch.nn as nn

class Net(nn.Module):
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
            skew: float = 0.3
            ):
        super(Net, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.distribution = distribution
        self.alpha = alpha
        self.sensitive_feature = sensitive_feature
        self.sensitive_value = sensitive_value
        self.skew = skew
        self.optimizer = None
        self.criterion = None