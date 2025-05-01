#!/usr/bin/env python3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# The command you want to run
base_command = ["./main.py", *sys.argv[1:]]

# How many parallel runs you want
num_runs = 10
total_done = 0


def run_command():
    global total_done
    try:
        result = subprocess.run(
            base_command,
            text=True,
            capture_output=True,
            check=True
        )
        total_done += 1
        print(f"Total done: {total_done}/{num_runs}", end="\r")
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


def main():
    print(f"Total done: 0/{num_runs}", end="\r")

    outputs = []
    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [executor.submit(run_command) for _ in range(num_runs)]
        for future in as_completed(futures):
            outputs.append(future.result())

    # Print all outputs
    for idx, output in enumerate(outputs, 1):
        print(f"\n--- Output #{idx} ---\n{output}\n")

if __name__ == "__main__":
    main()
