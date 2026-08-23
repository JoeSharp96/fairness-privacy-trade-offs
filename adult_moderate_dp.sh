#!/bin/bash
OUTPUT_PATH="$(pwd)/output/"
DIR_NAME="$(basename "$0")_$(date +"%s")"

NUM_SERVER_ROUNDS=30
FRACTION_EVALUATE=0.2
FRACTION_TRAIN=0.2
SAVE_MODEL=true
DATASET="adult"
BATCH_SIZE=128
MIN_PARTITION_SIZE=128
MAX_PHYSICAL_BATCH_SIZE=128
ALPHA=(0.2 0.5 1.0 10.0 500.0)
EPOCHS=1
LEARNING_RATE=0.001
DP=true
EPSILON=5.0
DELTA=1e-5
MAX_GRAD_NORM=1.0
NODES=5
SKEW=(0.05 0.1 0.2 0.3)
SENSITIVE_FEATURE='sex'
SENSITIVE_VALUE='Female'
TARGET_FEATURE='income'
SEED=(1 2 3 4 5 6 7 8 9 10)

# Tracking
TOTAL_RUNS=$((${#SEED[@]}*${#SKEW[@]}*${#ALPHA[@]}))
CURRENT_RUN=0

if ${SAVE_MODEL}; then
    if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir output
    fi
    mkdir output/${DIR_NAME}
    mkdir output/${DIR_NAME}/non_dp
    for epsilon in "${EPSILON[@]}"; do
        mkdir output/${DIR_NAME}/${epsilon}
    done
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
toml set --toml-path pyproject.toml tool.flwr.app.config.sensitive-value ${SENSITIVE_VALUE}
toml set --toml-path pyproject.toml --to-bool tool.flwr.app.config.dp-enabled ${DP}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.learning-rate ${LEARNING_RATE}
toml set --toml-path pyproject.toml tool.flwr.app.config.out-dir ${DIR_NAME}
toml set --toml-path pyproject.toml tool.flwr.app.config.target-feature ${TARGET_FEATURE}

toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.delta ${DELTA}
toml set --toml-path pyproject.toml --to-float tool.flwr.app.config.max-grad-norm ${MAX_GRAD_NORM}
toml set --toml-path pyproject.toml --to-int tool.flwr.app.config.max-physical-batch-size ${MAX_PHYSICAL_BATCH_SIZE}

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
    mv output/${DIR_NAME}/${seed} output/${DIR_NAME}/${EPSILON}
done

python reporting.py "$OUTPUT_PATH$DIR_NAME" "${SKEW[@]}" "${SEED[@]}" "${ALPHA[@]}"