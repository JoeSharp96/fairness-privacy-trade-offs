import sys
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs='+', required=True)
parser.add_argument("--skews", nargs="+", required=True)
parser.add_argument("--alphas", nargs="+", required=True)
parser.add_argument("--path", required=True)
args = parser.parse_args()

SKEW = args.skews
SEEDS = list(map(int, args.seeds))
ALPHA = args.alphas
PATH = args.path
YLABELS = {"server_eval_acc": "Acc", "server_demographic_parity": "STD", "server_equalised_odds": "EOD", "server_equal_opportunity": "EOP", "server_equalised_accuracy": "EA", "server_maj_accuracy": "Maj Acc", "server_min_accuracy": "Min Acc"}
directories = os.listdir(PATH)
directories.remove('results')

def aggregate_results()->None:
    """Aggregates results from output json file and stores data in a csv file. File is saved to the output folder"""
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
    """Mean for each metric is calculated from each seed."""
    for dir in directories:
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

def get_min_max_values(dataframes: list[DataFrame])->tuple[int,int]:
    """Gets the minimum and maximum value from to limit y axis in plot"""
    min_val = pd.concat(dataframes).groupby(level=0).min().min().min()
    max_val = pd.concat(dataframes).groupby(level=0).max().max().max()
    return min_val, max_val

def get_plot_data(metrics: list)->dict:
    """Gets the plot data from the mean value csv. Gets plot metadata from the results json."""
    plot_data = {metric:{"dir":{}} for metric in metrics}
    for metric in metrics:
        dfs = []
        for dir in directories:
            df = pd.read_csv(f"{PATH}/{dir}/{metric}_mean.csv", index_col=0, header=0)
            dfs.append(df)
            with open(f"{PATH}/{dir}/{SEEDS[0]}/{SKEW[0]}/results.json", "r") as fp:
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
    """Plots a grouped bar chart to show group accuracy differences under different epsilon budgets."""
    keys = list(plot_data.keys())
    index = [float(i) for i in ALPHA]
    string_values = [value for value in plot_data[keys[0]]['dir'].keys()]
    if "non_dp" in string_values and len(string_values) > 1:
        string_values.remove("non_dp")
        float_values = [float(value) for value in string_values]
        float_values.sort(reverse=True)
        epsilon_values = ["non_dp"] + [str(value) for value in float_values]
    elif "non_dp" not in string_values:
        float_values = [float(value) for value in string_values]
        float_values.sort(reverse=True)
        epsilon_values = [str(value) for value in float_values]
    else:
        epsilon_values = string_values
    df = pd.DataFrame(index=index)
    for key in keys:
        for e in epsilon_values:
            col_name = key + '_' + e
            e_df = plot_data[key]['dir'][e]['df']
            df[col_name] = e_df['0.3']
    df.plot.bar(figsize=(5.0,5.0), layout="compressed")
    plt.xlabel(u"\u03B1", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.ylim(0.77, 0.95)
    plt.minorticks_on()
    plt.grid(which='major', lw=0.5)
    plt.grid(which='minor', axis='y', lw=0.25, linestyle='--')
    plt.tick_params(which='major', labelsize=8)
    plt.tick_params(which='minor', bottom=False)
    plt.tight_layout()
    plt.legend(loc="best", ncols=2, fontsize=7)
    plt.savefig(f"{PATH}/results/disparate_impact.png", dpi=300)
    return

def plot_accuracy(metrics: list)->None:
    """Plots the accuracy """
    plot_data = get_plot_data(metrics)
    skews = [SKEW[0], SKEW[-1]]
    fig, axs = plt.subplots(len(metrics),len(skews), figsize=(5.0,4.4), layout="constrained")
    for i, metric in enumerate(metrics):
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for j, skew in enumerate(skews):
                axs[i][j].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[i][j].set_title(f"{float(skew)*100}% Female", fontsize=9)
                axs[i][j].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=9)
                axs[i][j].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=9)
                axs[i][j].minorticks_on()
                axs[i][j].grid(which='minor', axis="y", lw=0.25, linestyle='--')
                axs[i][j].grid(which='major', lw=0.5)
                axs[i][j].tick_params(which='major', labelsize=8)
                axs[i][j].tick_params(which='minor', bottom=False)
                axs[i][j].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0][0].legend(loc='best', fontsize=7, mode="expand", ncols=3)
    fig.savefig(f"{PATH}/results/dpvnondp.png", dpi=300)
    plot_disparate_impact(plot_data)
    return

def plot_eod(metrics: list)->None:
    plot_data = get_plot_data(metrics)
    fig, axs = plt.subplots(len(metrics),len(SKEW), figsize=(10.0,2.2), layout="constrained", squeeze=False)
    for metric in metrics:
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for i, skew in enumerate(SKEW):
                axs[0,i].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[0,i].set_title(f"{float(skew)*100}% Female", fontsize=9)
                axs[0,i].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=9)
                axs[0,i].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=9)
                axs[0,i].minorticks_on()
                axs[0,i].grid(which='minor', axis="y", lw=0.25, linestyle='--')
                axs[0,i].grid(which='major', lw=0.5)
                axs[0,i].tick_params(which='major', labelsize=8)
                axs[0,i].tick_params(which='minor', bottom=False)
                axs[0,i].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0,0].legend(loc='best', fontsize=7, mode="expand",ncols=3)
    fig.savefig(f"{PATH}/results/eod.png", dpi=300)
    return

def plot_std(metrics: list)->None:
    plot_data = get_plot_data(metrics)
    fig, axs = plt.subplots(len(metrics),len(SKEW), figsize=(10.0,2.2), layout="constrained", squeeze=False)
    for metric in metrics:
        for key in plot_data[metric]['dir'].keys():
            data = plot_data[metric]['dir'][key]
            for i, skew in enumerate(SKEW):
                axs[0,i].plot(ALPHA, data['df'][skew], label=data['label'])
                axs[0,i].set_title(f"{float(skew)*100}% Female", fontsize=9)
                axs[0,i].set_xlabel(plot_data[metric]['plot_labels']['xlabel'], fontsize=9)
                axs[0,i].set_ylabel(plot_data[metric]['plot_labels']['ylabel'], fontsize=9)
                axs[0,i].minorticks_on()
                axs[0,i].grid(which='minor', axis="y", lw=0.25, linestyle='--')
                axs[0,i].grid(which='major', lw=0.5)
                axs[0,i].tick_params(which='major', labelsize=8)
                axs[0,i].tick_params(which='minor', bottom=False)
                axs[0,i].set_ylim(plot_data[metric]['plot_labels']['ylim_min'] - 0.005, plot_data[metric]['plot_labels']['ylim_max'] + 0.005)
    axs[0,0].legend(loc='best', fontsize=7, mode="expand", ncols=3)
    fig.savefig(f"{PATH}/results/std.png", dpi=300)
    return

def plot_confusion_matrix()->None:
    """Creates and saves the confusion matrix for each model"""
    for dir in directories:
        with open(f"{PATH}/{dir}/{SEEDS[0]}/{SKEW[-1]}/results.json", "r") as fp:
            data = json.load(fp)
        final_data = data["final_metrics"]
        for alpha in [ALPHA[0],ALPHA[-1]]:
            min_cm = np.array([[final_data["server_min_tn"][alpha],final_data["server_min_fp"][alpha]], [final_data["server_min_fn"][alpha],final_data["server_min_tp"][alpha]]])
            maj_cm = np.array([[final_data["server_maj_tn"][alpha],final_data["server_maj_fp"][alpha]], [final_data["server_maj_fn"][alpha],final_data["server_maj_tp"][alpha]]])
            total_cm = np.add(min_cm, maj_cm)
            for cm, file in zip([min_cm, maj_cm, total_cm], ["min", "maj", "total"]):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["income < $50k","income >= $50k"])
                disp.plot(cmap=plt.cm.Blues)
                plt.tight_layout()
                disp.figure_.savefig(f"{PATH}/results/{file}_{dir}_{alpha}_{SKEW[-1]}_cm.png")
                plt.close()
    return

aggregate_results()
mean_results()
plot_accuracy(['server_maj_accuracy', 'server_min_accuracy'])
plot_std(['server_demographic_parity'])
plot_eod(['server_equalised_odds'])
plot_confusion_matrix()
