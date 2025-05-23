#!/usr/bin/env bash

# ==============================================================================
# This script performs a raw benchmark of a proposed defense system prompt on
# various models by utilizing the benign behaviors dataset.
# ==============================================================================

./main.py --mode benign --target-model gpt-3.5-turbo --benchmark-output-file ./results/gpt_3_5_defense_test_benign.json --log-level debug --target-model-system-prompt target-model-defense --dataset-path ./datasets/benign_behaviors.csv --raw-benchmark
clear
./main.py --mode benign --target-model gpt-4o --benchmark-output-file ./results/gpt_4o_defense_test_benign.json --log-level debug --target-model-system-prompt target-model-defense --dataset-path ./datasets/benign_behaviors.csv --raw-benchmark
clear
./main.py --mode benign --target-model vicuna --benchmark-output-file ./results/vicuna_defense_test_benign.json --log-level debug --target-model-system-prompt target-model-defense --dataset-path ./datasets/benign_behaviors.csv --raw-benchmark
clear
./main.py --mode benign --target-model llama2 --benchmark-output-file ./results/llama_2_defense_test_benign.json --log-level debug --target-model-system-prompt target-model-defense --dataset-path ./datasets/benign_behaviors.csv --raw-benchmark
