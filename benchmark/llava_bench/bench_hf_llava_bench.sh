#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./questions.jsonl \
    --image-folder ./images \
    --answers-file ./answers_hf.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
