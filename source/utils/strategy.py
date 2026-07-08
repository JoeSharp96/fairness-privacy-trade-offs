from flwr.app import MetricRecord

def get_individual_metrics(replies):
        """Get individual client loss and accuracy from train_replies for fairness metrics calculation."""
        client_loss = []
        client_acc = []
        for reply in replies:
            client_loss.append(reply.content["metrics"]["eval_loss"])
            client_acc.append(reply.content["metrics"]["eval_acc"])
        client_metrics = {"client_losses": client_loss, "client_acc": client_acc}
        return MetricRecord(client_metrics)