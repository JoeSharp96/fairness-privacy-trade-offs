"""pytorchexample: A Flower / PyTorch app."""

import torch
import random
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from opacus import PrivacyEngine

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check for personalised parameters. If none found, intialise with global model parameters.
    if "personalised_params" not in context.state:
        context.state["personalised_params"] = ArrayRecord(model.state_dict())
    # Store this rounds global params to bound personalised model.
    global_params = model.parameters()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    min_partition_size = context.run_config["min-partition-size"]
    alpha = context.run_config["alpha"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, alpha, min_partition_size)

    max_physical_batch_size = context.run_config["max-physical-batch-size"]
    lr = context.run_config['learning-rate']
    epochs = context.run_config["local-epochs"]
    train_loss = train_fn(
        model,
        trainloader,
        epochs,
        lr,
        device,
        max_physical_batch_size,
        context
    )

    # Construct and return reply Message
    # Save updated global model parameters.
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    # Train personalised model with Ditto if enabled.
    personalised_model = Net()
    personalised_model.load_state_dict(context.state["personalised_params"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    personalised_model.to(device)

    ditto_variables = {
        "lr": 0.001,
        "lambda": 0.01,
        "s": 5
    }

    _ = train_fn(
        model,
        trainloader,
        ditto_variables["s"],
        lr,
        device,
        max_physical_batch_size,
        context,
        ditto_variables,
        global_params
    )
    context.state["personalised_params"] = ArrayRecord(personalised_model.state_dict())

    return Message(content=content, reply_to=msg)
"""
    # Privacy hyperparams
    epsilon = context.run_config["epsilon"]
    delta = context.run_config["delta"]
    max_grad_norm = context.run_config["max-grad-norm"]
    max_physical_batch_size = context.run_config["max-physical-batch-size"]
    lr = msg.content["config"]["lr"]
    epochs = context.run_config["local-epochs"]
    # Privacy engine
    privacy_engine = PrivacyEngine()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs = epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max-grad-norm
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        epochs,
        lr,
        device,
        optimizer,
        max_physical_batch_size,
        epsilon,
        delta,
        privacy_engine
    )"""




@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    min_partition_size = context.run_config["min-partition-size"]
    alpha = context.run_config["alpha"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, alpha, min_partition_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    # Test personalised model
    if "personalised_params" in context.state:
        personalised_model = Net()
        personalised_model.load_state_dict(context.state["personalised_params"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        personalised_model.to(device)

        p_eval_loss, p_eval_acc = test_fn(
            personalised_model,
            valloader,
            device,
        )

        print(p_eval_acc)   


    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
