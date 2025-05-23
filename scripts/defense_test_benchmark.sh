#!/usr/bin/env bash

# ==============================================================================
# This script tests a proposed defense system prompt on various models and
# specifically for the historical role-play (hist-rp) approach. Additionally,
# the misuse behaviors dataset is utilized during this benchmark.
# ==============================================================================

./main.py --mode hist-rp --target-model gpt-3.5-turbo --benchmark-output-file ./results/gpt_3_5_defense_test.json --log-level debug --target-model-system-prompt target-model-defense --iters 3
clear
./main.py --mode hist-rp --target-model gpt-4o --benchmark-output-file ./results/gpt_4o_defense_test.json --log-level debug --target-model-system-prompt target-model-defense --iters 3
clear
./main.py --mode hist-rp --target-model vicuna --benchmark-output-file ./results/vicuna_defense_test.json --log-level debug --target-model-system-prompt target-model-defense --iters 3 --benchmark-workers 2
clear
./main.py --mode hist-rp --target-model llama2 --benchmark-output-file ./results/llama_2_defense_test.json --log-level debug --target-model-system-prompt target-model-defense --iters 3 --benchmark-workers 2
