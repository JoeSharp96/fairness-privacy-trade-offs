import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist
import source.models.femnist as femnist
import source.models.adult as adult

class Client:
    def __init__(self, model, ditto_model, partition_id):
        self.model = model
        self.ditto_model = ditto_model
        self.partition_id = partition_id
        self.trainloader = None
        self.testloader = None
    def load_data(ABC):
        return

class AdultClient(Client):
    def __init__(self, partition_id, lr, epochs, batch_size, num_partitions, distribution, alpha, ditto):
        self.model = adult.Adult(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        if ditto:
            self.ditto_model = adult.Adult(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        else:
            self.ditto_model = None
        super().__init__(self.model, self.ditto_model, partition_id)

    def load_data(self):
        self.trainloader, self.testloader = adult.load_data(self.partition_id, self.model.num_partitions, self.model.batch_size, self.model.alpha, self.model.distribution)

class FashionMnistClient(Client):
    def __init__(self, partition_id, lr, epochs, batch_size, num_partitions, distribution, alpha, ditto):
        self.model = fashion_mnist.FashionMnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        if ditto:
            self.ditto_model = fashion_mnist.FashionMnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        else:
            self.ditto_model = None
        super(FashionMnistClient, self).__init__(self.model, self.ditto_model, partition_id)

    def load_data(self):
        self.trainloader, self.testloader = fashion_mnist.load_data(self.partition_id, self.model.num_partitions, self.model.batch_size, self.model.alpha, self.model.distribution)

class MnistClient(Client):
    def __init__(self, partition_id, lr, epochs, batch_size, num_partitions, distribution, alpha, ditto):
        self.model = mnist.Mnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        if ditto:
            self.ditto_model = mnist.Mnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        else:
            self.ditto_model = None
        super().__init__(self.model, self.ditto_model, partition_id)

    def load_data(self):
        self.trainloader, self.testloader = mnist.load_data(self.partition_id, self.model.num_partitions, self.model.batch_size, self.model.alpha, self.model.distribution)

class FemnistClient(Client):
    def __init__(self, partition_id, lr, epochs, batch_size, num_partitions, distribution, alpha, ditto):
        self.model = femnist.Femnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        if ditto:
            self.ditto_model = femnist.Femnist(lr, epochs, batch_size, num_partitions, distribution, alpha, ditto = ditto)
        else:
            self.ditto_model = None
        super().__init__(self.model, self.ditto_model, partition_id)

    def load_data(self):
        self.trainloader, self.testloader = femnist.load_data(self.partition_id, self.model.num_partitions, self.model.batch_size, self.model.alpha, self.model.distribution)


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
        
    elif str.lower(dataset) == 'femnist':
        if train:
            return femnist.train, femnist.load_data
        else:
            return femnist.test, femnist.load_data

    elif str.lower(dataset) == 'adult':
        if train:
            return adult.train, adult.load_data
        else:
            return adult.test, adult.load_data
        
def get_client(dataset) -> Client:
    if str.lower(dataset) == 'mnist':
        return MnistClient
    
    elif str.lower(dataset) == 'fashion_mnist':
        return FashionMnistClient
        
    elif str.lower(dataset) == 'femnist':
        return FemnistClient

    elif str.lower(dataset) == 'adult':
        return AdultClient