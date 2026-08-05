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
SKEW = ["0.2", "0.3"]
SEEDS = [14, 15]
ALPHA = ["10.0", "100.0"]


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
    for seed in SEEDS:
        metrics ={}
        for skew in SKEW:
            with open(f"{PATH}/{seed}/{skew}/results.json") as fp:
                data = json.load(fp)
            for metric in data['final_metrics'].keys():
                if metric not in metrics:
                    metrics[metric] = {}
                results = data['final_metrics'][metric].values()
                metrics[metric][skew] = results
        for metric in metrics.keys():
            df = pd.DataFrame.from_dict(metrics[metric], orient='columns')
            df['alpha'] = ALPHA
            df.set_index(['alpha'], inplace=True)
            df.to_csv(f"{PATH}/{seed}/{metric}.csv")

def mean_results():
    for metric in ("acc","dp","eod","eop", "ea"):
        df_list = []
        for seed in SEEDS:
            df = pd.read_csv(f"{PATH}/{seed}/{metric}.csv", index_col=0)
            df_list.append(df)
        mean_df = pd.concat(df_list).groupby(level=0).mean()
        std_df = pd.concat(df_list).groupby(level=0).std()
        mean_df.to_csv(f"{PATH}/{metric}_mean.csv")
        std_df.to_csv(f"{PATH}/{metric}_std.csv")

def plot_results():
    mean_list = []
    std_list=[]
    for metric in ("acc","dp","eo"):
        mean_df = pd.read_csv(f"{PATH}/{metric}_mean.csv", index_col=0)
        mean_list.append(mean_df)
        std_df = pd.read_csv(f"{PATH}/{metric}_std.csv", index_col=0)
        std_list.append(std_df)
    fig, axs = plt.subplots(1,3, figsize=(16, 4))
    fig.suptitle("Adult")
    
    for mean_df, std_df, metric, i in zip(mean_list, std_list, ["Accuracy", "Demographic Parity", "Equalised Odds"], range(3)):
        for skew in SKEW:
            axs[i].plot(ALPHA, mean_df[skew], label=skew)
            axs[i].fill_between(ALPHA, mean_df[skew]-std_df[skew], mean_df[skew]+std_df[skew], alpha=0.5)
        axs[i].set_title(metric)
        axs[i].set_xlabel("\u03B1")
        y_label = "Acc" if metric == "Accuracy" else "Difference"
        axs[i].set_ylabel(y_label)
        axs[i].legend(title="Minority Skew")
    fig.savefig(f"{PATH}/eval_acc.jpg")

aggregate_results()
mean_results()
#plot_results()