#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

python "$SCRIPT_DIR/bench_traverse_tree.py" --spec-steps 1 --spec-topk 4 --spec-draft-tokens 4
python "$SCRIPT_DIR/bench_traverse_tree.py" --spec-steps 5 --spec-topk 4 --spec-draft-tokens 16
python "$SCRIPT_DIR/bench_traverse_tree.py" --spec-steps 5 --spec-topk 4 --spec-draft-tokens 64
