#!/bin/bash
OUTPUT_PATH="$(pwd)/output/"
DIR_NAME="$(basename "$0")_$(date +"%s")"
# Tune Hyperparameters
# FedAvg MNIST Non-IID Non-DP
# CONSTANTS
NUM_SERVER_ROUNDS=50
FRACTION_EVALUATE=0.2
FRACTION_TRAIN=0.2
SAVE_MODEL=true
DATASET="compas"
BATCH_SIZE=16
MIN_PARTITION_SIZE=16
ALPHA=(0.5)
EPOCHS=1
LEARNING_RATE=0.05
DITTO=false
NODES=5
# Due to a low amount of data in this particular dataset, I can only drop to 0.05 skew and 0.5 alpha.
SKEW=(0.05)
SENSITIVE_FEATURE='race:Caucasian'
SENSITIVE_VALUE=1.0
# 1996 24 33 258 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
SEED=(42)

# Tracking
TOTAL_RUNS=$((${#SEED[@]}*${#SKEW[@]}*${#ALPHA[@]}))
CURRENT_RUN=0

if ${SAVE_MODEL}; then
    if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir output
    fi
    mkdir output/${DIR_NAME}
fi

toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.num-server-rounds ${NUM_SERVER_ROUNDS}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.num-partitions ${NODES}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-evaluate ${FRACTION_EVALUATE}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.fraction-train ${FRACTION_TRAIN}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.save-model ${SAVE_MODEL}
toml set --toml-path pyproject.toml tool.flwr.app.config.dataset ${DATASET}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.batch-size ${BATCH_SIZE}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.local-epochs ${EPOCHS}
toml set --toml-path pyproject.toml tool.flwr.app.config.sensitive-feature ${SENSITIVE_FEATURE}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.sensitive-value ${SENSITIVE_VALUE}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.dp-enabled ${DP}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.learning-rate ${LEARNING_RATE}
toml set --toml-path pyproject.toml tool.flwr.app.config.out-dir ${DIR_NAME}

# If using G4 GPU, set cpu to 12 and gpu to 0.25
flwr federation simulation-config --num-supernodes ${NODES} --client-resources-num-cpus 2 --client-resources-num-gpus 0.0
for seed in "${SEED[@]}"; do
    toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.seed ${seed}
    if ${SAVE_MODEL}; then
        mkdir output/${DIR_NAME}/${seed}
    fi
    for skew in "${SKEW[@]}"; do
        toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.skew ${skew}
        for alpha in "${ALPHA[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN+1))
            toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.alpha ${alpha}
            echo "Starting Run ${CURRENT_RUN}/${TOTAL_RUNS}:
            Seed: ${seed} | Skew: ${skew} | Alpha: ${alpha}"
            flwr run . --stream
        done
    done
done

python reporting.py "$OUTPUT_PATH$DIR_NAME" ${NUM_SERVER_ROUNDS} ${DP} "${SKEW}" "${SEED}" "${ALPHA}"