#!/bin/bash

export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

python3 -m sglang.benchmark.dspark_sps_profiler all \
    --base-url http://127.0.0.1:30100 \
    --batch-size 1 2 4 8 10 \
    --fracs 0.01 0.2 0.4 0.6 0.8 1.0 \
    --repeats 2 \
    --out /home/l00993641/sglang/scripts/dspark_graph_additive.json \
    --no-plot