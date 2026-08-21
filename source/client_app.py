"""Flower Client"""

import torch
import gc
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from source.utils.client import Client


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data.
    Return local model paramerters and training loss."""
    # Load the model and initialize it with the received weights
    client = Client(
        partition_id=context.node_config["partition-id"],
        lr=context.run_config['learning-rate'],
        epochs=context.run_config["local-epochs"],
        batch_size=context.run_config["batch-size"],
        num_partitions=context.node_config["num-partitions"],
        distribution=context.run_config["distribution"],
        alpha=context.run_config["alpha"],
        sensitive_feature=context.run_config["sensitive-feature"],
        sensitive_value=context.run_config["sensitive-value"],
        skew=context.run_config["skew"],
        seed=context.run_config["seed"],
        target_feature=context.run_config["target-feature"],
        dataset=context.run_config["dataset"]
    )
    # Load weights to the client model
    client.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Move model to GPU if applicable.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client.model.to(device)

    # Load federated dataset partition. If this is the first training round, the client will save a copy of the data sample.
    client.load_data(context.run_config["out-dir"])

    # Train model on local dataset
    train_loss = client.fit(device, msg.content["config"])

    # Construct Flower ArrayRecord to store updated model parameters and MetricRecord store training metrics. 
    model_record = ArrayRecord(client.model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(client.trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    # Client's are stateless and created each round. However, some issues with memory leakage have been occuring. At the end of each round, client is delete and cleared.
    del client
    gc.collect()

    # Return metrics and updated weights
    return Message(content=content, reply_to=msg)




@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    client = Client(
        partition_id=context.node_config["partition-id"],
        lr=context.run_config['learning-rate'],
        epochs=context.run_config["local-epochs"],
        batch_size=context.run_config["batch-size"],
        num_partitions=context.run_config["num-partitions"],
        distribution=context.run_config["distribution"],
        alpha=context.run_config["alpha"],
        sensitive_feature=context.run_config["sensitive-feature"],
        sensitive_value=context.run_config["sensitive-value"],
        skew=context.run_config["skew"],
        seed=context.run_config["seed"],
        target_feature=context.run_config["target-feature"],
        dataset=context.run_config["dataset"]
    )
    client.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client.load_data(context.run_config["out-dir"])

    # Call the evaluation function
    eval_loss, eval_acc = client.test(device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(client.testloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    del client
    gc.collect()
    return Message(content=content, reply_to=msg)