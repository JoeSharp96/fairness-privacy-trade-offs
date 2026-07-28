
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# Create int that counts number of directories in the dir.
# Each run gets it's own file, then keep everything else the same
# Pass in already generated output path. Then extend path and mkdir with new value.
# Save the run_config.
def output_dir(config):
    """Create directory for output graph and data"""
    path = Path.cwd() / f"output/{config['out-dir']}/{config['skew']}"
    if not os.path.exists(path):
        print("hello")
        path.mkdir(parents=True, exist_ok=False)
        with open(f"{path}/results.json","w",encoding="utf-8") as fp:
            results = {
                "run_metrics":{},
                "final_metrics":{
                    "acc":{},
                    "minority_acc":{},
                    "majority_acc":{},
                    "dp":{},
                    "eo":{}
                },
                "run_config": config
                }
            json.dump(results,fp)
    return path

def save_metrics(result, save_path, rounds, alpha, loss_disparity=None, acc_disparity=None, ditto=False):
    """Save metrics to output directory as JSON file"""
    results = {"disparity": {"loss_disparity": loss_disparity, "acc_disparity": acc_disparity},
               "round_metrics": {},
               "ditto_metrics": {}}
    for i in range(1,rounds+1):
        train_metrics = dict(result.train_metrics_clientapp.get(i,{}))
        eval_client_metrics = dict(result.evaluate_metrics_clientapp.get(i,{}))
        eval_server_metrics = dict(result.evaluate_metrics_serverapp.get(i,{}))
        round_result = {
            "train_loss": train_metrics["train_loss"],
            "eval_client_loss": eval_client_metrics["eval_loss"],
            "eval_client_acc": eval_client_metrics["eval_acc"],
            "eval_server_loss": eval_server_metrics["loss"],
            "eval_server_acc": eval_server_metrics["accuracy"],
            "eval_server_min_acc": eval_server_metrics["minority_accuracy"],
            "eval_server_maj_acc": eval_server_metrics["majority_accuracy"],
            "demographic_parity": eval_server_metrics["demographic_parity"],
            "equalised_odds": eval_server_metrics["equalised_odds"]
        }
        results["round_metrics"][i] = round_result
        
        if ditto:
            ditto_result = {
                "ditto_train_loss": train_metrics["ditto_train_loss"],
                "ditto_eval_client_loss": eval_client_metrics["ditto_eval_loss"],
                "ditto_eval_client_acc": eval_client_metrics["ditto_eval_acc"]
            }
            results["ditto_metrics"][i] = ditto_result

    data={}
    if os.path.exists(f"{save_path}/results.json"):
        with open(f"{save_path}/results.json", "r", encoding="utf-8") as fp:
            data = json.load(fp)

    with open(f"{save_path}/results.json", "w", encoding="utf-8") as fp:
        data["run_metrics"][alpha] = results
        data["final_metrics"]["acc"][alpha] = results["round_metrics"][rounds]["eval_server_acc"]
        data["final_metrics"]["minority_acc"][alpha] = results["round_metrics"][rounds]["eval_server_min_acc"]
        data["final_metrics"]["majority_acc"][alpha] = results["round_metrics"][rounds]["eval_server_maj_acc"]
        data["final_metrics"]["dp"][alpha] = results["round_metrics"][rounds]["demographic_parity"]
        data["final_metrics"]["eo"][alpha] = results["round_metrics"][rounds]["equalised_odds"]
        json.dump(data, fp)
    

def save_graphs(save_path, rounds, alpha):
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
