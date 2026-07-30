"""Flower Server"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from source.utils.reporting import output_dir, save_metrics, save_graphs
from source.utils.server import get_fl_strategy, get_functions
from source.utils.models import get_model
import source.models.adult as adult
import source.models.compas as compas
import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist
import source.models.femnist as femnist

# Create ServerApp
app = ServerApp()

class ServerConfig:
    def __init__(self, model: adult.Adult | compas.Compas, dataset: str, seed: int):
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.testloader = None
    
    def load_data(self):
        if str.lower(self.dataset) == "adult":
            self.testloader = adult.load_centralized_dataset(
                num_partitions=self.model.num_partitions,
                batch_size=self.model.batch_size, 
                alpha=self.model.alpha,
                sensitive_feature=self.model.sensitive_feature,
                sensitive_value=self.model.sensitive_value,
                skew=self.model.skew,
                seed=self.seed
                )
        elif str.lower(self.dataset) == "compas":
            self.testloader = compas.load_centralized_dataset(
                num_partitions=self.model.num_partitions,
                batch_size=self.model.batch_size, 
                alpha=self.model.alpha,
                sensitive_feature=self.model.sensitive_feature,
                sensitive_value=self.model.sensitive_value,
                skew=self.model.skew,
                seed=self.seed
                )
        elif str.lower(self.dataset) == "mnist":
            self.testloader = mnist.load_centralized_dataset(self.model.distribution, self.model.batch_size)
        elif str.lower(self.dataset) == "fashion_mnist":
            self.testloader = fashion_mnist.load_centralized_dataset(self.model.distribution, self.model.batch_size)
        elif str.lower(self.dataset) == "femnist":
            self.testloader = femnist.load_centralized_dataset(self.model.distribution, self.model.batch_size)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model and intialise parameters
    Net = get_model(context.run_config["dataset"])
    config = context.run_config
    # !! Net() initialization has a magic number for partition_size. I think this is because it doesn't automatically pass in...
    # I should create a variable in the shell script that is used to set the number nodes and partitions.
    global_model = Net(
        lr=config["learning-rate"],
        epochs=config["local-epochs"],
        batch_size=config["batch-size"], 
        num_partitions=config["num-partitions"], 
        distribution=config["distribution"], 
        alpha=config["alpha"], 
        sensitive_feature=config["sensitive-feature"],
        sensitive_value=config["sensitive-value"],
        skew=config["skew"]
        )
    arrays = ArrayRecord(global_model.state_dict())
    server = ServerConfig(global_model, config["dataset"], config["seed"])

    # Initialize FL strategy
    strategy, train_config = get_fl_strategy(config)

    # Start strategy for `num_rounds`
    result, individual_metrics = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        server=server,
        train_config=ConfigRecord(train_config),
        evaluate_config=ConfigRecord({"ditto": context.run_config["ditto"], "dataset": context.run_config["dataset"], "distribution": context.run_config["distribution"]}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
        fraction_malicious=context.run_config["fraction-malicious"]
    )

    # Record fairness metrics
    loss_disparity, acc_disparity = get_disparity(
        individual_metrics,
        result.evaluate_metrics_clientapp[num_rounds]["eval_acc"],
        result.evaluate_metrics_clientapp[num_rounds]["eval_loss"]
        )

    # Save model
    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        save_path = output_dir(config=context.run_config)
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"{save_path}/final_model.pt")
        save_metrics(result, save_path, num_rounds, context.run_config["alpha"], loss_disparity, acc_disparity, context.run_config["ditto"])
        save_graphs(save_path,num_rounds,context.run_config["alpha"])
    # Clear GPU cache at the end of each run
    torch.cuda.empty_cache()
    
    

def global_evaluate(server_round: int, arrays: ArrayRecord, server: ServerConfig) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    server.model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    server.model.to(device)
    # Load entire test set
    if server.testloader is None:
        server.load_data()

    # Evaluate the global model on the test set
    print(server.testloader)
    test_loss, test_acc, test_dp, test_eo, test_min_acc, test_maj_acc = server.model.test(server.testloader, device, True)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss, "demographic_parity": test_dp, "equalised_odds": test_eo, "minority_accuracy": test_min_acc, "majority_accuracy": test_maj_acc})

def get_disparity(individual_eval_metrics, agg_eval_acc, agg_eval_loss):
    """Calculate loss and accuracy disparity of global model across client local data."""
    num_clients = len(individual_eval_metrics['client_losses'])
    ld = 0.0
    ad = 0.0
    for i in range(num_clients):
        ld += (individual_eval_metrics['client_losses'][i] - agg_eval_loss) ** 2
        ad += (individual_eval_metrics['client_acc'][i] - agg_eval_acc) ** 2
    ld = ld / num_clients
    ad = ad / num_clients
    return ld, ad
