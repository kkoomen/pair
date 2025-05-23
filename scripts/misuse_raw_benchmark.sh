#!/usr/bin/env bash

# ==============================================================================
# This script runs the raw benchmark for misuse behaviors on various models.
# Raw benchmarks means no attack model is employed during this benchmark.
# ==============================================================================

./main.py --target-model gpt-3.5-turbo --raw-benchmark --benchmark-output-file ./results/gpt_3_5.json --log-level debug
clear
./main.py --target-model gpt-4o --raw-benchmark  --benchmark-output-file ./results/gpt_4o.json --log-level debug
clear
./main.py --target-model vicuna --raw-benchmark --benchmark-workers 2 --benchmark-output-file results/vicuna.json --log-level debug
clear
./main.py --target-model llama2 --raw-benchmark --benchmark-workers 2 --benchmark-output-file results/llama2.json --log-level debug
