#!/bin/bash
OUTPUT_PATH="$(pwd)/output/"
DIR_NAME="$(basename "$0")_$(date +"%s")"
# Tune Hyperparameters
# FedAvg MNIST Non-IID Non-DP
# CONSTANTS
STRATEGY="fedavg"
NUM_SERVER_ROUNDS=2
FRACTION_EVALUATE=0.1
FRACTION_TRAIN=0.1
FRACTION_MALICIOUS=0.0
SAVE_MODEL=true
DATASET="mnist"
DISTRIBUTION="non-iid"
BATCH_SIZE=16
MIN_PARTITION_SIZE=16
ALPHA=0.5
EPOCHS=1
LEARNING_RATE=(0.001 0.005 0.01 0.05 0.1 0.5)
DITTO=false
DP=false

if ${SAVE_MODEL}; then
    if [ -d "$OUTPUT_PATH" ]; then
        mkdir output
    fi
    mkdir output/${DIR_NAME}
fi

toml set --toml-path pyproject.toml tool.flwr.app.config.strategy ${STRATEGY}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.num-server-rounds ${NUM_SERVER_ROUNDS}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-evaluate ${FRACTION_EVALUATE}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-train ${FRACTION_TRAIN}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-malicious ${FRACTION_MALICIOUS}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.save-model ${SAVE_MODEL}
toml set --toml-path pyproject.toml tool.flwr.app.config.dataset ${DATASET}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.batch-size ${BATCH_SIZE}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.alpha ${ALPHA}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.local-epochs ${EPOCHS}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.ditto ${DITTO}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.dp-enabled ${DP}
toml set --toml-path pyproject.toml tool.flwr.app.config.out-dir ${DIR_NAME}

for lr in "${LEARNING_RATE[@]}"; do
    toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.learning-rate ${lr}
    flwr run . --stream
done

python reporting.py "$OUTPUT_PATH/$DIR_NAME" ${NUM_SERVER_ROUNDS} ${DP}