import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist

def get_functions(dataset, train=True):
    """Returns train, test and dataloading funtions depending on the selected dataset"""
    if str.lower(dataset) == 'mnist':
        if train:
            return mnist.train, mnist.load_data
        else:
            return mnist.test, mnist.load_data
    
    elif str.lower(dataset) == 'fashion_mnist':
        if train:
            return fashion_mnist.train, fashion_mnist.load_data
        else:
            return fashion_mnist.test, fashion_mnist.load_data
