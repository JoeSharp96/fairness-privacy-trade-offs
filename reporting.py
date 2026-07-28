import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = sys.argv[1]
ROUNDS = int(sys.argv[2])
DP = bool(sys.argv[3])
SKEW = [0.01, 0.05, 0.1, 0.2, 0.3]


def tune_fedavg():
    return

def tune_qfedavg():
    """Saves the best performing hyperparameters"""

def plot_accuracy_with_dp():
    """"""
    plt.figure(figsize=(5,5))
    for dir in os.listdir(PATH):
        print(dir)
        with open(f"{PATH}/{dir}/results.json", "r") as jsonfile:
            data = json.load(jsonfile)
            df = pd.DataFrame.from_dict(data["round_metrics"], orient="index")
        
        with open(f"{PATH}/{dir}/run_config.json","r") as jsonfile:
            config = json.load(jsonfile)

        epochs = config['local-epochs']
        if config['dp-enabled']:
            epsilon = config['epsilon']
            text = f"Sever rounds = {ROUNDS}\nLocal epochs = {epochs}\nε = {epsilon}"
        plt.plot(df.index, df['eval_server_acc'], label=config['clipping-mode'])  
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Local DP Accuracy')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend(title="Clipping Method")
    plt.savefig(f"{PATH}/eval_acc.jpg")
    return

def plot_accuracy():
    """Creates matplotlib graphs of results and saves them as JPG files"""
    plt.figure(figsize=(5,5))
    for dir in os.listdir(PATH):
        print(dir)
        with open(f"{PATH}/{dir}/results.json", "r") as jsonfile:
            data = json.load(jsonfile)
            df = pd.DataFrame.from_dict(data["round_metrics"], orient="index")
        
        with open(f"{PATH}/{dir}/run_config.json","r") as jsonfile:
            config = json.load(jsonfile)

        epochs = config['local-epochs']
        if config['dp-enabled']:
            epsilon = config['epsilon']
            text = f"Sever rounds = {ROUNDS}\nLocal epochs = {epochs}\nε = {epsilon}"
        else:
            text = f"Sever rounds = {ROUNDS}\nLocal epochs = {epochs}\nNon-DP"
        plt.plot(df.index, df['eval_server_acc'], label=config['clipping-mode'])  
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Local DP Accuracy')
    plt.text(0,0.85,text)
    plt.grid(True)
    plt.legend(title="Clipping Method")
    plt.savefig(f"{PATH}/eval_acc.jpg")

def aggregate_results():
    metrics ={}
    for skew in SKEW:
        with open(f"{PATH}/{skew}/results.json") as fp:
            data = json.load(fp)
        for metric in data['final_metrics'].keys():
            if metric not in metrics:
                metrics[metric] = {}
            results = data['final_metrics'][metric].values()
            metrics[metric][skew] = results
    columns = data['final_metrics']['acc'].keys()
    for metric in metrics.keys():
        print(metrics[metric])
        df = pd.DataFrame.from_dict(metrics[metric], orient='index', columns=columns)
        df.to_csv(f"{PATH}/{metric}.csv")

aggregate_results()