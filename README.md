# Privacy, Fairness and Accuracy Trade-offs in Federated Learning
## Description

## Prerequisites
The following dependencies are required to run the project files:
- flwr[\simulation]
- flwr_datasets[\vision]
- torch
- torchvision
- opacus
- toml-cli
- scikit-learn
## Installation via terminal
Use the pip package manager to install project dependencies:
```bash
pip install git+https://github.com/JoeSharp96/fairness-privacy-trade-offs
```
If installing from GitHub, you can clone the repository here with the command:
```bash
git clone https://github.com/JoeSharp96/fairness-privacy-trade-offs.git
```
After installing, you may need to reset your machine before proceeding.
## Usage
This project uses Flower to simulate federated learning. The Flower framework is capable of simulating multiple actors in parallel, depending on the number of resources each client is given access to.

You can specify the number of available CPUs and portion of GPU to each actor with the following command:

```bash
flwr federation simulation-config --client-resources-num-gpus num-gpus --client-resources-num-cpus num-cpus
```
Replace ```num-gpus``` with the a float value representing the percentage of the GPU each client should use (```0.25``` represents 25%).

Replace ```num-cpus``` with the an integer value representing the number of CPU cores each client should use. 

The total actors that run in parallel is equal to min(SYS_CPUS/num_cpus, SYS_GPUS/num_gpus). Where ```SYS_CPUS``` is the total number of CPU cores available, and ```SYS_GPUS``` is the total number of GPUs available.

By defualt, the simulation config is set to:
```bash
flwr federation simulation-config --client-resources-num-gpus 0.0 --client-resources-num-cpus 2
```
## Parameters
Adujusting the parameters for the simulation can be done updating the specific shell script. Increase or decreasing the ```ALPHA```, ```SKEW```, ```SEED``` variables will change the number of runs exponentially.

## Run via terminal
Before starting, move into the project file as the current working directory:
```bash
cd fairness-privacy-trade-offs
```

There are three simulations that can be run:
- Non-DP
- Strict DP
- Moderate DP

Each simulation can be run individually or all together. Note that if run individually, plots created will only display simulated model. To view all models in the same plot, use the ```adult_all.sh``` shell script.

All simulations can be run as the same time with the following shell command:

```bash
bash adult_all.sh
```

Each can be run using the following shell scripts:
```bash
bash adult_non_dp.sh
bash adult_moderate_dp.sh
bash adult_strict_dp.sh
```
### Run via VSCode & Google Colab Notebooks
To quickly run the model in VSCode or Google Colab, you can use the ```vscode_notebook.ipynb``` and ```colab_notebook.ipynb``` files. Run all the cells until the "Running the federated learning model" section to install all files and dependencies.


## Outputs
Metrics and plots can be found in the ```outputs``` directory. This will be created on the first run of a simulation.

Collected metrics include:
- Global Accuracy
- Group Accuracy (Minority and Majority)
- Confusion Matrix for both minority and majority groups
- Equalised Odds Difference
- Statistical Parity Difference

## Files
### ```server_app.py```
Flower server app, intialises federated learning model and implements strategy.

### ```client_app.py```
Flower client app. Calls client training methods and returns updated parameters.

### ```utils/client.py```
Contains a client object. In this file you can find the methods to load the federated dataset, skew data, test and train.

### ```utils/server.py```
Contains the server config object. Contains methods to load test data, run global testing, generate fairness metrics.

### ```utils/reporting.py```
Saves the metrics from training and testing. Saves dataset samples.