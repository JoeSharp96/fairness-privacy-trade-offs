"""Flower Client"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from source.utils.client import get_functions,get_client,Client,FashionMnistClient
from source.utils.models import get_model


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    # Loads dataset model. Need to find a better way of doing this as it's pretty inefficient. Gets called every round for every client.
    # Issue is, each client app is a unqiue instance, each round generates a new set
    ClientClass = get_client(msg.content["config"]["dataset"])
    client = ClientClass(
        partition_id=context.node_config["partition-id"],
        lr=context.run_config['learning-rate'],
        epochs=context.run_config["local-epochs"],
        batch_size=context.run_config["batch-size"],
        num_partitions=10,
        distribution=context.run_config["distribution"],
        alpha=context.run_config["alpha"],
        ditto=msg.content["config"]["ditto"]
    )
    client.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client.model.to(device)
    print(msg.content["config"]["is_malicious"])

    if msg.content["config"]["ditto"]:
        # Store this rounds global params to bound personalised model.
        client.ditto_model.global_params = client.model.parameters()
        client.ditto_model.lmbda = msg.content["config"]["ditto_lambda"]
        client.ditto_model.lr = msg.content["config"]["ditto_lr"]
        client.ditto_model.epochs = msg.content["config"]["ditto_local_epochs"]
        # Check for personalised parameters. If none found, save intitial params.
        if "ditto_params" not in context.state:
            context.state["ditto_params"] = ArrayRecord(client.ditto_model.state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    min_partition_size = context.run_config["min-partition-size"]
    alpha = context.run_config["alpha"]
    distribution = context.run_config["distribution"]
    client.load_data()

    lr = context.run_config['learning-rate']
    epochs = context.run_config["local-epochs"]
    train_loss = client.fit(client.model, device, msg.content["config"])

    # Construct and return reply Message
    # Save updated global model parameters.
    model_record = ArrayRecord(client.model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(client.trainloader.dataset),
    }

    if client.ditto_model is not None:
        client.ditto_model.load_state_dict(context.state["ditto_params"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        client.ditto_model.to(device)

        ditto_train_loss = client.fit(
            client.ditto_model, 
            device, 
            msg.content["config"],
            ditto=True
        )
        
        # Save updated personalised model parameters.
        context.state["ditto_params"] = ArrayRecord(client.ditto_model.state_dict())
        metrics["ditto_train_loss"] = ditto_train_loss

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)




@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    ClientClass = get_client(msg.content["config"]["dataset"])
    client = ClientClass(
        partition_id=context.node_config["partition-id"],
        lr=context.run_config['learning-rate'],
        epochs=context.run_config["local-epochs"],
        batch_size=context.run_config["batch-size"],
        num_partitions=10,
        distribution=context.run_config["distribution"],
        alpha=context.run_config["alpha"],
        ditto=msg.content["config"]["ditto"]
    )
    client.model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client.model.to(device)


    # Load the data
    client.load_data()

    # Call the evaluation function
    eval_loss, eval_acc = client.model.test(
        client.testloader,
        device
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(client.testloader.dataset),
    }

    # Test personalised model
    if client.ditto_model is not None:
        if "ditto_params" not in context.state:
            context.state["ditto_params"] = ArrayRecord(client.ditto_model.state_dict())
        client.ditto_model.load_state_dict(context.state["ditto_params"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        client.ditto_model.to(device)

        ditto_eval_loss, ditto_eval_acc = client.ditto_model.test(
            client.testloader,
            device
        ) 
        metrics["ditto_eval_loss"] = ditto_eval_loss
        metrics["ditto_eval_acc"] = ditto_eval_acc
        

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)