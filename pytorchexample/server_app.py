"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    print(context.run_config)
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    # Fraction_train determines how many possible nodes will be used in training.
    strategy = FedAvg(fraction_evaluate=fraction_evaluate, fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
    
    # Pretty sure this is my script to produce graphs. Make this it's own function.
    agg_acc = []
    print(result)
    for round in result.evaluate_metrics_clientapp.values():
        agg_acc.append(round['eval_acc'])

    epochs = context.run_config['local-epochs']
    if context.run_config['dp-enabled']:
        epsilon = context.run_config['epsilon']
        text = f"Sever rounds = {num_rounds}\nLocal epochs = {epochs}\nε = {epsilon}"
    else:
        text = f"Sever rounds = {num_rounds}\nLocal epochs = {epochs}\nNon-DP"

    plt.figure(figsize=(5, 5))
    plt.plot(pd.DataFrame(np.array(agg_acc)), marker='o', color='b', label='Round Accuracy')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Round')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend()
    plt.savefig("output.jpg")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
