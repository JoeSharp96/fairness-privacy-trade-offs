import torch.nn as nn
import pytorchexample.models.mnist as mnist
import pytorchexample.models.fashion_mnist as fashion_mnist

def get_model(dataset) -> nn.Module:
    """Returns model for specified dataset."""
    if str.lower(dataset) == 'mnist':
        return mnist.Net
    elif str.lower(dataset) == 'fashion_mnist':
        return fashion_mnist.Net
