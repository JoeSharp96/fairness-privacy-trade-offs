# Privacy, Fairness and Accuracy Trade-offs in Federated Learning
## Description
This is the description

## Prerequisites
The following dependencies are required to run the project files:
- flwr[\simulation]
- flwr_datasets[\vision]
- torch
- torchvision
- opacus
- toml-cli
## Installation
Use the pip package manager to install project dependencies:
```bash
pip install git+https://github.com/JoeSharp96/fairness-privacy-trade-offs
```
## Usage
The Flower framework is capable of simulating multiple actors in parallel, depending on the number of resources each client is given access to.

You can specify the number of available CPUs and portion of GPU to each actor with the following command:

```bash
flwr federation simulation-config --client-resources-num-gpus num-gpus --client-resources-num-cpus num-cpus
```
Replace ```num-gpus``` with the a float value representing the percentage of the GPU each client should use (```0.25``` represents 25%).

Replace ```num-cpus``` with the an integer value representing the number of CPU cores each client should use. 

The total actors that run in parallel is equal to min(SYS_CPUS/num_cpus, SYS_GPUS/num_gpus).

By defualt, the simulation config is set to:
```bash
flwr federation simulation-config --client-resources-num-gpus 0.0 --client-resources-num-cpus 2
```

There are three simulations that can be run:
- Non-DP
- Strict DP
- Moderate DP

Each can be run using the following shell scripts:
```bash
bash adult_non_dp.sh
bash adult_moderate_dp.sh
bash adult_strict_dp.sh
```

## Outputs
Metrics and outputs can be found in the 'outputs' directory. This will be created on the first run of a simulation.

Collected metrics include:
- Global Accuracy
- Group Accuracy (Minority and Majority)
- Confusion Matrix for both minority and majority groups
- Equalised Odds Difference
- Equal Opportunity Difference
- Demographic Parity Difference
