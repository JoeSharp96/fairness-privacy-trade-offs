#!/bin/bash
OUTPUT_PATH="$(pwd)/output/"
DIR_NAME="$(basename "$0")_$(date +"%s")"
# Tune Hyperparameters
# FedAvg MNIST Non-IID Non-DP
# CONSTANTS
STRATEGY="fedavg"
NUM_SERVER_ROUNDS=100
FRACTION_EVALUATE=0.2
FRACTION_TRAIN=0.2
FRACTION_MALICIOUS=0.0
SAVE_MODEL=true
DATASET="adult"
DISTRIBUTION="non-iid"
BATCH_SIZE=16
MIN_PARTITION_SIZE=16
ALPHA=(0.1 0.5 1.0 10.0 100.0)
EPOCHS=1
LEARNING_RATE=(0.05)
DITTO=false
DP=false
NODES=5
SKEW=(0.01 0.05 0.1 0.2 0.3)
SENSITIVE_FEATURE='sex'
SENSITIVE_VALUE='female'

if ${SAVE_MODEL}; then
    if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir output
    fi
    mkdir output/${DIR_NAME}
fi

toml set --toml-path pyproject.toml tool.flwr.app.config.strategy ${STRATEGY}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.num-server-rounds ${NUM_SERVER_ROUNDS}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.num-partitions ${NODES}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-evaluate ${FRACTION_EVALUATE}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-train ${FRACTION_TRAIN}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-malicious ${FRACTION_MALICIOUS}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.save-model ${SAVE_MODEL}
toml set --toml-path pyproject.toml tool.flwr.app.config.dataset ${DATASET}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.batch-size ${BATCH_SIZE}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.local-epochs ${EPOCHS}
toml set --toml-path pyproject.toml tool.flwr.app.config.sensitive-feature ${SENSITIVE_FEATURE}
toml set --toml-path pyproject.toml tool.flwr.app.config.sensitive-value ${SENSITIVE_VALUE}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.ditto ${DITTO}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.dp-enabled ${DP}
toml set --toml-path pyproject.toml tool.flwr.app.config.out-dir ${DIR_NAME}

flwr federation simulation-config --num-supernodes ${NODES} --client-resources-num-cpus 2 --client-resources-num-gpus 0.0

for skew in "${SKEW[@]}"; do
    toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.skew ${skew}
    for alpha in "${ALPHA[@]}"; do
        toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.alpha ${alpha}
        flwr run . --stream
    done
done

python reporting.py "$OUTPUT_PATH$DIR_NAME" ${NUM_SERVER_ROUNDS} ${DP}