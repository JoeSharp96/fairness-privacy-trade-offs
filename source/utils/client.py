
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
import source.models.net as Net
import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist
import source.models.femnist as femnist
import source.models.adult as adult

class Client:
    def __init__(self, model: Net, ditto_model: Net, partition_id: int):
        self.model = model
        self.ditto_model = ditto_model
        self.partition_id = partition_id
        self.trainloader = None
        self.testloader = None

    def load_data(ABC):
        return
    
    def ditto_train(self, model):
        """Bound the personalised model updates to not drift too far from the global model."""
        with torch.no_grad():
            for p, g_p, in zip(model.parameters(), model.global_params):
                update = p - model.lr * (p.grad + model.lmbda * torch.dist(p, g_p, p=2))
                p.copy_(update)
    
    def train(self, model: Net, x_key, y_key, trainloader, device, epochs, ditto):
        running_loss = 0.0
        model.train()
        for _ in range(epochs):
            for batch in trainloader:
                model.optimizer.zero_grad()
                images = batch[x_key].to(device)
                labels = batch[y_key].to(device)
                loss = model.criterion(model(images), labels)
                loss.backward()
                # Train ditto model
                if ditto:
                    self.ditto_train(model)
                model.optimizer.step()
                running_loss += loss.item()
        return running_loss

    
    def fit_with_dp(self, model, train_config):
        privacy_engine = PrivacyEngine()
        model, model.optimizer, dp_trainloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=model.optimizer,
            data_loader=self.trainloader,
            epochs = model.epochs,
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
        return model, model.optimizer, dp_trainloader
    
    def fit(self, model: Net, device, train_config, ditto):
        """Train the model on the training set."""
        epochs = model.epochs
        # DP Enabled training
        if train_config["dp"] and train_config["dp_mode"] == "local" and ditto == False:
            model, model.optimizer, dp_trainloader = self.fit_with_dp(model, train_config)
            with BatchMemoryManager(
                data_loader=dp_trainloader, 
                max_physical_batch_size=train_config["dp_max_physical_batch_size"], 
                optimizer=model.optimizer
            ) as memory_safe_data_loader:
                running_loss = self.train(model, 'image', 'label', memory_safe_data_loader, device, epochs, ditto)
        else:
            running_loss = self.train(model, 'image', 'label', self.trainloader, device, epochs, ditto)
        avg_trainloss = running_loss / (epochs * len(self.trainloader))
        return avg_trainloss

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

    def train(self, model: Net, trainloader, device, epochs, ditto):
        running_loss = 0.0
        model.train()
        for _ in range(epochs):
            for X_batch, y_batch in trainloader:
                model.optimizer.zero_grad()
                outputs = model(X_batch).to(device)
                loss = model.criterion(outputs, y_batch).to(device)
                loss.backward()
                if ditto:
                    self.ditto_train(model)
                model.optimizer.step()
                running_loss += loss.item()
        return running_loss

    def fit(self, model, device, train_config, ditto=False):
        epochs=model.epochs
        model.to(device)  # move model to GPU if available
        model.criterion = torch.nn.BCELoss().to(device)
        model.optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
        if train_config["dp"] and train_config["dp_mode"] == "local" and ditto == False:
            model, model.optimizer, dp_trainloader = self.fit_with_dp(model, train_config)
            with BatchMemoryManager(
                data_loader=dp_trainloader,
                max_physical_batch_size=train_config["dp_max_physical_batch_size"],
                optimizer=model.optimizer
            ) as memory_safe_data_loader:
                running_loss = self.train(model, memory_safe_data_loader, device, epochs, ditto)
        else:
            running_loss = self.train(model, self.trainloader, device, epochs, ditto)
        avg_trainloss = running_loss / (epochs * len(self.trainloader))
        return avg_trainloss

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

    def fit(self, model: Net, device, train_config, ditto=False):
        model.to(device)
        model.criterion = torch.nn.CrossEntropyLoss().to(device)
        model.optimizer = torch.optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
        return super().fit(model,device,train_config,ditto)

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

    def fit(self, model: Net, device, train_config, ditto=False):
        model.to(device)
        model.criterion = torch.nn.CrossEntropyLoss().to(device)
        model.optimizer = torch.optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
        return super().fit(model,device,train_config,ditto)

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

    def fit(self, model: Net, device, train_config, ditto=False):
        model.to(device)
        model.criterion = torch.nn.CrossEntropyLoss().to(device)
        model.optimizer = torch.optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
        return super().fit(model,device,train_config,ditto)

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
