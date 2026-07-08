from source.strategies.custom import CustomFedAvg
from source.strategies.customqfedavg import CustomQFedAvg
from source.strategies.boundedclipping import DifferentialPrivacyServerSideBoundedClipping
from source.strategies.customdpfixedclip import CustomDifferentialPrivacyFixedClipping
from source.strategies.customdpadaptiveclip import CustomDifferentialPrivacyAdaptiveClipping
from opacus.accountants.utils import get_noise_multiplier
import source.models.mnist as mnist
import source.models.fashion_mnist as fashion_mnist

def get_fl_strategy(run_config):
    """Returns FL strategy and training config"""
    fl_strategy = None

    # Initialise ditto and dp flags to control training flow.
    train_config = {"ditto": run_config["ditto"], "dp": run_config["dp-enabled"], "dp_mode": run_config["dp-mode"], "dp_clipping": run_config["clipping-mode"], "dataset": run_config["dataset"]}
    
    # Add ditto parameters to train_config if ditto is enabled
    if train_config["ditto"]:
        train_config["ditto_lr"] = run_config["ditto-lr"]
        train_config["ditto_lambda"] = run_config["ditto-lambda"]
        train_config["ditto_local_epochs"] = run_config["ditto-local-epochs"]
    
    # Select chosen strategy
    if run_config["strategy"] == 'qfedavg':
        fl_strategy = CustomQFedAvg(
            client_learning_rate=run_config["learning-rate"],
            q=run_config["q"],
            fraction_evaluate=run_config["fraction-evaluate"],
            fraction_train=run_config["fraction-train"],
            train_loss_key = "train_loss",
            min_available_nodes= run_config["min-available-nodes"]
            )
        
    elif run_config["strategy"] == 'fedavg':
        fl_strategy = CustomFedAvg(
            fraction_evaluate=run_config["fraction-evaluate"],
            fraction_train=run_config["fraction-train"],
            min_available_nodes= run_config["min-available-nodes"]
            )
        
    else:
        raise ValueError(f"Invalid strategy: {run_config['strategy']}. Must be 'fedavg' or 'qfedavg'")
    
    # If DP flag is true, add DP configurations to train config wrap fl_strategy with dp_strategy
    if train_config["dp"]:
        dp_attributes = {
            "dp_max_grad_norm": run_config["max-grad-norm"],
            "dp_max_physical_batch_size": run_config["max-physical-batch-size"],
            "dp_epsilon": run_config["epsilon"],
            "dp_delta": run_config["delta"],
            "dp_min_bound": 1e-5
        }
        # If local DP is chosen, original fl_strategy is returned with configuration for Opacus
        fl_strategy, train_config = get_dp_strategy(fl_strategy, run_config, dp_attributes, train_config)
    return fl_strategy, train_config
    
def get_dp_strategy(strategy, run_config, attributes, train_config):
    """Return Custom DP strategy (if necessary) and training config"""
    
    # Flower doesn't automatically calculate the noise_multiplier from a target epsilon and delta. Opacus has this functionality. 
    if run_config["dp-mode"] == "server":
        noise_multiplier = get_noise_multiplier(
            target_epsilon=run_config["epsilon"],
            target_delta=run_config["delta"],
            sample_rate=run_config["fraction-train"],
            steps=run_config["num-server-rounds"],
        )
    
    # Return strategy and update training config dict.
    if run_config["clipping-mode"] == "flat":
        if run_config["dp-mode"] == "server":
            return CustomDifferentialPrivacyFixedClipping(
                strategy=strategy,
                noise_multiplier=noise_multiplier,
                num_sampled_clients=run_config["min-available-nodes"] * run_config["fraction-train"],
                clipping_norm=run_config["max-grad-norm"]
                ), train_config
        
        elif run_config["dp-mode"] == "local":
            train_config = add_local_dp_config(train_config, attributes)
            return strategy, train_config
        
        else:
            raise ValueError(f"Invalid dp value: {run_config['dp-mode']}. Must be 'server' or 'local'.")
        
    elif run_config['clipping-mode'] == "adaptive":
        if run_config["dp-mode"] == "server":
            return CustomDifferentialPrivacyAdaptiveClipping(
                strategy=strategy,
                noise_multiplier=noise_multiplier,
                num_sampled_clients=run_config["min-available-nodes"] * run_config["fraction-train"]
                ), train_config
        
        elif run_config["dp-mode"] == "local":
            train_config = add_local_dp_config(train_config, attributes)
            return strategy, train_config
        
        else:
            raise ValueError(f"Invalid dp value: {run_config['dp-mode']}. Must be 'server' or 'local'.")
        
    elif run_config["clipping-mode"] == "bounded":
        if run_config["dp-mode"] == "server":
            return DifferentialPrivacyServerSideBoundedClipping(
                strategy=strategy,
                noise_multiplier=noise_multiplier,
                num_sampled_clients=run_config["min-available-nodes"] * run_config["fraction-train"],
                min_bound=run_config["clipping-lower-bound"]
                )
        
        elif run_config["dp-mode"] == "local":
            train_config["dp_clipping"] = "adaptive"
            train_config["dp_min_bound"] = run_config["clipping-lower-bound"]
            train_config = add_local_dp_config(train_config, attributes)
            return strategy, train_config
        
        else:
            raise ValueError(f"Invalid dp value: {run_config['dp-mode']}. Must be 'server' or 'local'.")
        
    elif run_config["clipping-mode"] == "automatic":
        if run_config["dp-mode"] == "server":
            return
        
        elif run_config["dp-mode"] == "local":
            return
        
        else:
            raise ValueError(f"Invalid dp value: {run_config['dp-mode']}. Must be 'server' or 'local'.")
        
    else:
        raise ValueError(f"Invalid clipping value: {run_config['clipping-mode']}. Must be 'flat', 'adaptive', 'bounded', or 'automatic'.")
    
def add_local_dp_config(train_config, attributes):
        """Add all required DP variables to train_config dict"""
        for key, value in attributes.items():
            train_config[key] = value
        return train_config

def get_functions(dataset):
    if str.lower(dataset) == 'mnist':
        return mnist.test, mnist.load_centralized_dataset
    if str.lower(dataset) == 'fashion_mnist':
        return fashion_mnist.test, fashion_mnist.load_centralized_dataset