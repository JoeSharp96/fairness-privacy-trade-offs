"""Flower Client"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from source.utils.client import get_functions
from source.utils.models import get_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    # Loads dataset model. Need to find a better way of doing this as it's pretty inefficient. Gets called every round for every client.
    # Issue is, each client app is a unqiue instance, each round generates a new set
    Net = get_model(msg.content["config"]["dataset"])
    train_fn, load_data = get_functions(msg.content["config"]["dataset"])

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if msg.content["config"]["ditto"]:
        # Store this rounds global params to bound personalised model.
        global_params = model.parameters()
        ditto_model = Net()
        # Check for personalised parameters. If none found, save intitial params.
        if "ditto_params" not in context.state:
            context.state["ditto_params"] = ArrayRecord(ditto_model.state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    min_partition_size = context.run_config["min-partition-size"]
    alpha = context.run_config["alpha"]
    distribution = context.run_config["distribution"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, alpha, min_partition_size, distribution)

    lr = context.run_config['learning-rate']
    epochs = context.run_config["local-epochs"]
    train_loss = train_fn(
        model,
        trainloader,
        epochs,
        lr,
        device,
        msg.content["config"]
    )

    # Construct and return reply Message
    # Save updated global model parameters.
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    if msg.content["config"]["ditto"]:
        ditto_model.load_state_dict(context.state["ditto_params"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ditto_model.to(device)

        ditto_train_loss = train_fn(
            ditto_model,
            trainloader,
            msg.content["config"]["ditto_local_epochs"],
            msg.content["config"]["ditto_lr"],
            device,
            msg.content["config"],
            global_params
        )
        
        # Save updated personalised model parameters.
        context.state["ditto_params"] = ArrayRecord(ditto_model.state_dict())
        metrics["ditto_train_loss"] = ditto_train_loss

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)




@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    Net = get_model(msg.content["config"]["dataset"])
    test_fn, load_data = get_functions(msg.content["config"]["dataset"], train=False)
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
    distribution = context.run_config["distribution"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, alpha, min_partition_size, distribution)

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
    if msg.content["config"]["ditto"]:
        ditto_model = Net()
        if "ditto_params" not in context.state:
            context.state["ditto_params"] = ArrayRecord(ditto_model.state_dict())
        ditto_model.load_state_dict(context.state["ditto_params"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ditto_model.to(device)

        ditto_eval_loss, ditto_eval_acc = test_fn(
            ditto_model,
            valloader,
            device,
        ) 
        metrics["ditto_eval_loss"] = ditto_eval_loss
        metrics["ditto_eval_acc"] = ditto_eval_acc
        

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
