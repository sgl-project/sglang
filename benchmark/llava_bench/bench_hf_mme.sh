#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./mme_pack/llava_mme_bench_replace.jsonl \
    --image-folder ./mme_pack/MME_Benchmark_release_version \
    --answers-file ./answers_hf_mme.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
