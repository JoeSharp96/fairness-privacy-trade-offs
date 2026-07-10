"""Flower Server"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from source.utils.reporting import output_dir, save_metrics, save_graphs
from source.utils.server import get_fl_strategy, get_functions
from source.utils.models import get_model

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    # Load global model and intialise parameters
    Net = get_model(context.run_config["dataset"])
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FL strategy
    strategy, train_config = get_fl_strategy(context.run_config)

    # Start strategy for `num_rounds`
    result, individual_metrics = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_config),
        evaluate_config=ConfigRecord({"ditto": context.run_config["ditto"], "dataset": context.run_config["dataset"], "distribution": context.run_config["distribution"]}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
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
        save_metrics(result, save_path, num_rounds, loss_disparity, acc_disparity, context.run_config["ditto"])
        save_graphs(save_path,num_rounds)
    
    

def global_evaluate(server_round: int, arrays: ArrayRecord, dataset: str, distribution = None) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    Net = get_model(dataset)
    test_fn, load_centralized_dataset_fn = get_functions(dataset)
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load entire test set
    test_dataloader = load_centralized_dataset_fn(distribution)

    # Evaluate the global model on the test set
    test_loss, test_acc = test_fn(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

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
