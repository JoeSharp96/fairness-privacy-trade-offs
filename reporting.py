import sys
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PATH = sys.argv[1]
ROUNDS = int(sys.argv[2])
DP = bool(sys.argv[3])

def plot_accuracy_with_dp():
    return

def plot_accuracy():
    """Creates matplotlib graphs of results and saves them as JPG files"""
    peak_acc = 0.0
    peak_lr = 0.0
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
        plt.plot(df.index, df['eval_server_acc'], label=config['learning-rate'])
        if df['eval_server_acc'].max() > peak_acc:
            peak_acc = df['eval_server_acc'].max()
            peak_lr = config['learning-rate']
        
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.text(0,0.85,text)
    plt.figtext(0,0, f"Optimal LR: {peak_lr}\nPeak Acc: {peak_acc}")
    plt.grid(True)
    plt.legend(title="Learning Rates")
    plt.savefig(f"{PATH}/eval_acc.jpg")



if DP:
    plot_accuracy_with_dp()
else:
    plot_accuracy()