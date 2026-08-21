import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay

PATH = sys.argv[1]
ROUNDS = int(sys.argv[2])
DP = bool(sys.argv[3])
SKEW = ["0.05", "0.1", "0.2", "0.3"]
SEEDS = [1,2,3,4,5,6,7,8,9,10]
ALPHA = ["0.2","0.5","1.0","10.0","500.0"]
YLABELS = {"server_eval_acc": "Acc", "server_demographic_parity": "STD", "server_equalised_odds": "EOD", "server_equal_opportunity": "EOP", "server_equalised_accuracy": "EA", "server_maj_accuracy": "Maj Acc", "server_min_accuracy": "Min Acc"}

def aggregate_results()->None:
    directories = os.listdir(PATH)
    for dir in directories:
        for seed in SEEDS:
            metrics ={}
            for skew in SKEW:
                with open(f"{PATH}/{dir}/{seed}/{skew}/results.json") as fp:
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
                df.to_csv(f"{PATH}/{dir}/{seed}/{metric}.csv")
    return

def mean_results()->None:
    for dir in os.listdir(PATH):
        for metric in ("server_eval_acc","server_demographic_parity","server_equalised_odds","server_equal_opportunity", "server_equalised_accuracy",'server_maj_accuracy', 'server_min_accuracy'):
            df_list = []
            for seed in SEEDS:
                df = pd.read_csv(f"{PATH}/{dir}/{seed}/{metric}.csv", index_col=0)
                df_list.append(df)
            mean_df = pd.concat(df_list).groupby(level=0).mean()
            std_df = pd.concat(df_list).groupby(level=0).std()
            mean_df.to_csv(f"{PATH}/{dir}/{metric}_mean.csv")
            std_df.to_csv(f"{PATH}/{dir}/{metric}_std.csv")
    return

def median_results()->None:
    for metric in ("server_demographic_parity","server_equalised_odds","server_equal_opportunity"):
        df_list = []
        for seed in SEEDS:
            df = pd.read_csv(f"{PATH}/{seed}/{metric}.csv", index_col=0)
            df_list.append(df)
        median_df = pd.concat(df_list).groupby(level=0).median()
        median_df.to_csv(f"{PATH}/{metric}_median.csv")
    return

def get_min_max_values(dataframes: list[DataFrame])->tuple[int,int]:
    min_val = pd.concat(dataframes).groupby(level=0).min().min().min()
    max_val = pd.concat(dataframes).groupby(level=0).max().max().max()
    return min_val, max_val

def get_plot_data(metrics: list)->dict:
    directories = os.listdir(PATH)
    plot_data = {metric:{"dir":{}} for metric in metrics}
    for metric in metrics:
        dfs = []
        for dir in directories:
            df = pd.read_csv(f"{PATH}/{dir}/{metric}_mean.csv", index_col=0, header=0)
            dfs.append(df)
            with open(f"{PATH}/{dir}/1/0.1/results.json", "r") as fp:
                run_dict = json.load(fp)
            if run_dict['run_config']['dp-enabled']:
                label = str(run_dict['run_config']['epsilon'])
            else:
                label = "Non-DP"
            plot = {
                "df": df,
                "label": label
            }
            plot_data[metric]['dir'][dir] = plot
        min_value, max_value = get_min_max_values(dfs)
        plot_data[metric]['plot_labels'] = {"ylabel": YLABELS[metric], "xlabel": "\u03B1", "ylim_min": min_value, "ylim_max": max_value}
    return plot_data

def plot_disparate_impact(plot_data: dict)->None:
    keys = list(plot_data.keys())
    index = [float(i) for i in ALPHA]
    epsilon_values = [value for value in plot_data[keys[0]]['dir'].keys()]
    epsilon_values = sorted(epsilon_values,reverse=True)
    df = pd.DataFrame(index=index)
    for key in keys:
        for e in epsilon_values:
            col_name = key + '_' + e
            e_df = plot_data[key]['dir'][e]['df']
            print(e_df['0.3'])
            df[col_name] = e_df['0.3']
    df.plot.bar()
    plt.xlabel("\u03B1")
    plt.ylabel("Accuracy")
    plt.grid(lw=0.5)
    plt.ylim(0.77, 0.95)
    plt.savefig(f"{PATH}/non_dp/disparate_impact.png", dpi=300)
    return

def plot_accuracy(metrics: list)->None:
    plot_data = get_plot_data(metrics)
    skews = [SKEW[0], SKEW[-1]]
    fig, axs = plt.subplots(len(metrics),len(skews), figsize=(5.0,4.4), layout="constrained")
    for i, metric in enumerate(metrics):
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for j, skew in enumerate(skews):
                axs[i][j].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[i][j].set_title(f"{float(skew)*100}% Female", fontsize=7)
                axs[i][j].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=6)
                axs[i][j].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=6)
                axs[i][j].grid(lw=0.5)
                axs[i][j].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0][0].legend(loc='best', fontsize=8, mode="expand")
    fig.savefig(f"{PATH}/non_dp/dpvnondp.png", dpi=300)
    plot_disparate_impact(plot_data)
    return

def plot_eod(metrics: list)->None:
    plot_data = get_plot_data(metrics)
    fig, axs = plt.subplots(len(metrics),len(SKEW), figsize=(10.0,2.2), layout="constrained")
    for metric in metrics:
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for i, skew in enumerate(SKEW):
                axs[i].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[i].set_title(f"{float(skew)*100}% Female", fontsize=7)
                axs[i].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=6)
                axs[i].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=6)
                axs[i].grid(lw=0.5)
                axs[i].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0].legend(loc='best', fontsize=8, mode="expand")
    fig.savefig(f"{PATH}/non_dp/eod.png", dpi=300)
    return

def plot_std(metrics: list)->None:
    plot_data = get_plot_data(metrics)
    fig, axs = plt.subplots(len(metrics),len(SKEW), figsize=(10.0,2.2), layout="constrained")
    for metric in metrics:
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for i, skew in enumerate(SKEW):
                axs[i].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[i].set_title(f"{float(skew)*100}% Female", fontsize=7)
                axs[i].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=6)
                axs[i].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=6)
                axs[i].grid(lw=0.5)
                axs[i].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0].legend(loc='best', fontsize=8, mode="expand")
    fig.savefig(f"{PATH}/non_dp/std.png", dpi=300)
    return

def plot_confusion_matrix()->None:
    """Creates and saves the confusion matrix for each model"""
    directories = os.listdir(PATH)
    for dir in directories:
        with open(f"{PATH}/{dir}/1/0.3/results.json", "r") as fp:
            data = json.load(fp)
        final_data = data["final_metrics"]
        for alpha in [ALPHA[0],ALPHA[1],ALPHA[-1]]:
            min_cm = np.array([[final_data["server_min_tn"][alpha],final_data["server_min_fp"][alpha]], [final_data["server_min_fn"][alpha],final_data["server_min_tp"][alpha]]])
            maj_cm = np.array([[final_data["server_maj_tn"][alpha],final_data["server_maj_fp"][alpha]], [final_data["server_maj_fn"][alpha],final_data["server_maj_tp"][alpha]]])
            total_cm = np.add(min_cm, maj_cm)
            for cm, file in zip([min_cm, maj_cm, total_cm], ["min", "maj", "total"]):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["income < $50k","income >= $50k"])
                disp.plot(cmap=plt.cm.Blues).figure_.savefig(f"{PATH}/non_dp/{file}_{dir}_{alpha}_0.3_cm.png")
                plt.close()
    return

plot_accuracy(['server_maj_accuracy', 'server_min_accuracy'])
#plot_std(['server_demographic_parity'])
#plot_eod(['server_equalised_odds'])
#plot_confusion_matrix()
