
from flwr.app import UserConfig
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def output_dir(config: UserConfig) -> tuple[Path, str]:
    """Create directory for output graph and data"""
    current_time = datetime.now()
    out_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    path = Path.cwd() / f"output/{out_dir}"
    path.mkdir(parents=True, exist_ok=False)

    with open(f"{path}/run_config.json","w",encoding="utf-8") as fp:
        json.dump(config,fp)

    return path

def save_metrics(result, save_path, rounds):
    """Save metrics"""
    results = {}
    for i in range(1,rounds+1):
        train_metrics = dict(result.train_metrics_clientapp.get(i,{}))
        eval_client_metrics = dict(result.evaluate_metrics_clientapp.get(i,{}))
        eval_server_metrics = dict(result.evaluate_metrics_serverapp.get(i,{}))
        round_result = {
            "train_loss": train_metrics["train_loss"],
            "eval_client_loss": eval_client_metrics["eval_loss"],
            "eval_client_acc": eval_client_metrics["eval_acc"],
            "eval_server_loss": eval_server_metrics["loss"],
            "eval_server_acc": eval_server_metrics["accuracy"]
        }
        results[i] = round_result
    
    with open(f"{save_path}/results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp)

def save_graphs(save_path, rounds):
    """Creates matplotlib graphs of results and saves them as JPG files"""
    with open(f"{save_path}/results.json", "r") as jsonfile:
        df = pd.read_json(jsonfile, orient="index")
        #results = json.load(jsonfile)
    
    with open(f"{save_path}/run_config.json","r") as jsonfile:
        config = json.load(jsonfile)

    epochs = config['local-epochs']
    if config['dp-enabled']:
        epsilon = config['epsilon']
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nε = {epsilon}"
    else:
        text = f"Sever rounds = {rounds}\nLocal epochs = {epochs}\nNon-DP"

    plt.figure(figsize=(5, 5))
    plt.plot(df.index, df['eval_client_acc'], marker='o', color='b', label='Aggregate Client Accuracy')
    plt.plot(df.index, df['eval_server_acc'], marker='x', color='r', label='Global Accuracy')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_path}/eval_acc.jpg")