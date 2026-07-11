import torch.nn as nn
import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist
import source.models.femnist as femnist
import source.models.adult as adult

# Create a base class for Client
# Add all basic methods

def get_model(dataset) -> nn.Module:
    """Returns model for specified dataset."""
    if str.lower(dataset) == 'mnist':
        return mnist.Mnist
    elif str.lower(dataset) == 'fashion_mnist':
        return fashion_mnist.FashionMnist
    elif str.lower(dataset) == 'femnist':
        return femnist.Femnist
    elif str.lower(dataset) == 'adult':
        return adult.Adult
