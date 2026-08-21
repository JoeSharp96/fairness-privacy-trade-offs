
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, UserConfig
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg, Result
from flwr_datasets import FederatedDataset

# Create int that counts number of directories in the dir.
# Each run gets it's own file, then keep everything else the same
# Pass in already generated output path. Then extend path and mkdir with new value.
# Save the run_config.
def get_path(dir: str, seed: int, skew: float) -> Path:
    """Creates path to output directory for current run and skew"""
    path = Path.cwd() / f"output/{dir}/{seed}/{skew}"
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    return path

def output_dir(config: UserConfig) -> Path:
    """Create directory for output graph and data"""
    path = get_path(config["out-dir"], config["seed"], config["skew"])
    if not os.path.exists(f"{path}/results.json"):
        with open(f"{path}/results.json","w",encoding="utf-8") as fp:
            results = {
                "run_metrics":{},
                "final_metrics":{},
                "run_config": config
                }
            json.dump(results,fp)
    return path

def save_metrics(
        result: Result, 
        save_path: Path, 
        rounds: int, 
        alpha: float
        )->None:
    """Save metrics to output directory as JSON file"""
    results = {"round_metrics": {}}
    for i in range(1,rounds+1):
        train_metrics = dict(result.train_metrics_clientapp.get(i,{}))
        eval_client_metrics = dict(result.evaluate_metrics_clientapp.get(i,{}))
        eval_server_metrics = dict(result.evaluate_metrics_serverapp.get(i,{}))
        round_result = {}
        for key, value in train_metrics.items():
            round_result[key] = value

        for key, value in eval_client_metrics.items():
            round_result[f"client_{key}"] = value

        for key, value in eval_server_metrics.items():
            round_result[f"server_{key}"] = value

        results["round_metrics"][i] = round_result

    data={}
    if os.path.exists(f"{save_path}/results.json"):
        with open(f"{save_path}/results.json", "r", encoding="utf-8") as fp:
            data = json.load(fp)

    with open(f"{save_path}/results.json", "w", encoding="utf-8") as fp:
        data["run_metrics"][alpha] = results
        for key, value in results["round_metrics"][rounds].items():
            if key not in data["final_metrics"]:
                data["final_metrics"][key] = {}
            data["final_metrics"][key][alpha] = value

        json.dump(data, fp)
    

def save_graphs(save_path: Path, rounds: int, alpha: float) -> None:
    """Creates matplotlib graphs of results and saves them as JPG files"""
    with open(f"{save_path}/results.json", "r") as jsonfile:
        data = json.load(jsonfile)
        df = pd.DataFrame.from_dict(data["run_metrics"][str(alpha)]["round_metrics"], orient="index")
        config = data["run_config"]

    epochs = config['local-epochs']
    if config['dp-enabled']:
        epsilon = config['epsilon']
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nε = {epsilon}"
    else:
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nNon-DP"

    plt.figure(figsize=(5, 5))
    plt.plot(df.index, df['client_eval_acc'], marker='o', color='b', label='Aggregate Client Accuracy')
    plt.plot(df.index, df['server_eval_acc'], marker='x', color='r', label='Global Accuracy')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_path}/{alpha}/eval_acc.jpg")

def save_partitions(
        sensitive_feature: str,
        sensitive_value: str,
        fds: FederatedDataset,
        seed: int,
        output_directory: str,
        num_partitions: int,
        skew: float,
        alpha: float
        )->None:
    """For each client, count how many grouped attributes there are. Save all data to a .csv file"""
    path=get_path(output_directory, seed, skew)
    path = path.joinpath(f"{alpha}")
    path.mkdir(parents=True, exist_ok=True)
    minority_count = []
    majority_count = []
    for i in range(num_partitions):
        dataset = fds.load_partition(i, "train").with_format("pandas")[:]
        minority_count.append(dataset.where(dataset[sensitive_feature]==sensitive_value).count()[sensitive_feature])
        majority_count.append(dataset.where(dataset[sensitive_feature]!=sensitive_value).count()[sensitive_feature])
    df = pd.DataFrame(data={"min":minority_count, "maj":majority_count})
    df.to_csv(f"{path}/data_sample.csv", mode="w", header=True, index=False)
