#!/usr/bin/env bash

# ==============================================================================
# This script runs the raw benchmark for benign behaviors on various models.
# Raw benchmarks means no attack model is employed during this benchmark.
# ==============================================================================

./main.py --mode benign --target-model gpt-3.5-turbo --raw-benchmark --benchmark-output-file ./results/gpt_3_5_benign.json --dataset-path ./datasets/benign_behaviors.csv --log-level debug
clear
./main.py --mode benign --target-model gpt-4o --raw-benchmark  --benchmark-output-file ./results/gpt_4o_benign.json --dataset-path ./datasets/benign_behaviors.csv --log-level debug
clear
./main.py --mode benign --target-model vicuna --raw-benchmark --benchmark-workers 2 --benchmark-output-file results/vicuna_benign.json --dataset-path ./datasets/benign_behaviors.csv --log-level debug
clear
./main.py --mode benign --target-model llama2 --raw-benchmark --benchmark-workers 2 --benchmark-output-file results/llama2_benign.json --dataset-path ./datasets/benign_behaviors.csv --log-level debug
