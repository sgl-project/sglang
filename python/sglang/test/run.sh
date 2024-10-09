#!/bin/bash

# Run few_shot_gsm8k.py 5 times sequentially
echo "Starting runs for few_shot_gsm8k.py..."
for i in {1..5}
do
    echo "Run #$i: few_shot_gsm8k.py"
    python few_shot_gsm8k.py
    # Optional: Add a delay between runs
    # sleep 1
done
echo "Completed runs for few_shot_gsm8k.py."

# Run few_shot_gsm8k_engine_async.py 5 times sequentially
echo "Starting runs for few_shot_gsm8k_engine_async.py..."
for i in {1..5}
do
    echo "Run #$i: few_shot_gsm8k_engine_async.py"
    python few_shot_gsm8k_engine_async.py
    # Optional: Add a delay between runs
    # sleep 1
done
echo "Completed runs for few_shot_gsm8k_engine_async.py."
