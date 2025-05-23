#!/usr/bin/env bash

# ==============================================================================
# This script runs the benchmark for misuse behaviors on various models for the
# following approaches:
# - Roleplay
# - Hist-Roleplay
# - Hist-Roleplay-NL
# - Hist-Roleplay-DE
# - Hist-Roleplay-FR
# ==============================================================================

./main.py --mode misuse --target-model gpt-3.5-turbo --benchmark-output-file ./results/gpt_3_5.json --log-level debug
clear
./main.py --mode misuse --target-model gpt-4o --benchmark-output-file ./results/gpt_4o.json --log-level debug
clear
./main.py --mode misuse --target-model vicuna --benchmark-workers 2 --benchmark-output-file results/vicuna.json --log-level debug
clear
./main.py --mode misuse --target-model llama2 --benchmark-workers 2 --benchmark-output-file results/llama2.json --log-level debug
