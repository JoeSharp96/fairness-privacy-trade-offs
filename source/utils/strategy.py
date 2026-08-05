from flwr.app import UserConfig
from source.strategies.custom import CustomFedAvg

def add_local_dp_config(train_config: dict, attributes: dict) -> dict:
        """Add all required DP variables to train_config dict"""
        for key, value in attributes.items():
            train_config[key] = value
        return train_config

def get_fl_strategy(run_config: UserConfig) -> tuple[CustomFedAvg, dict]:
    """Returns FL strategy and training config"""
    fl_strategy = None

    # Initialise ditto and dp flags to control training flow.
    train_config = {"dp": run_config["dp-enabled"], "dataset": run_config["dataset"]}

    fl_strategy = CustomFedAvg(
        fraction_train=run_config["fraction-train"],
        fraction_evaluate=run_config["fraction-evaluate"],
        min_available_nodes=run_config["min-available-nodes"]
    )
    
    # If DP flag is true, add DP configurations to train config wrap fl_strategy with dp_strategy
    if train_config["dp"]:
        dp_attributes = {
            "dp_max_grad_norm": run_config["max-grad-norm"],
            "dp_max_physical_batch_size": run_config["max-physical-batch-size"],
            "dp_epsilon": run_config["epsilon"],
            "dp_delta": run_config["delta"]
        }
        train_config = add_local_dp_config(train_config, dp_attributes)
    return fl_strategy, train_config