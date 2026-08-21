"""Flower Server"""
import torch
import gc
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from source.utils.reporting import output_dir, save_metrics, save_graphs
from source.utils.server import ServerConfig
from source.utils.strategy import get_fl_strategy
from source.models.adult import Adult
from source.models.compas import Compas
from source.utils.client import clear_fds

# Create ServerApp
app = ServerApp()

def get_model(dataset: str)->Adult|Compas:
    if dataset == 'adult':
        return Adult
    else:
        return Compas

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds = context.run_config["num-server-rounds"]
    config = context.run_config
    
    # Load global model and intialise parameters
    Net = get_model(context.run_config["dataset"])
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

    # Flower custom object ArrayRecord used to store global params and update params from clients.
    arrays = ArrayRecord(global_model.state_dict())
    server = ServerConfig(global_model, config["dataset"], config["seed"], config["target-feature"])
    
    # Initialize FL strategy, if DP settings are passed into the train_config
    strategy, train_config = get_fl_strategy(config)

    # Start strategy for `num_rounds`.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        server=server,
        train_config=ConfigRecord(train_config),
        evaluate_config=ConfigRecord({"dataset": context.run_config["dataset"]}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
    )

    # Save model
    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        save_path = output_dir(config=context.run_config)
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"{save_path}/{config['alpha']}/final_model.pt")
        save_metrics(result, save_path, num_rounds, context.run_config["alpha"])
        save_graphs(save_path,num_rounds,context.run_config["alpha"])
    # Clear GPU cache at the end of each run
    del server
    torch.cuda.empty_cache()
    clear_fds()
    # Clears server object to prevent memory leaks
    gc.collect()
    
    

def global_evaluate(server_round: int, arrays: ArrayRecord, server: ServerConfig) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    server.model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    server.model.to(device)
    # Check if the dataset has been loaded. If not, load the test set to the server config.
    if server.testloader is None:
        server.load_centralized_dataset()

    # Test current weights and return accuracy with the fairness metrics.
    metrics = server.test(device)

    # Return the metrics using MetricRecord.
    return MetricRecord(metrics)
