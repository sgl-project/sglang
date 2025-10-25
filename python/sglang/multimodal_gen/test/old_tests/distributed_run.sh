#!/bin/bash

num_gpus=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503 \
    tp_example.py



num_gpus=2
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503  \
    sgl-diffusion/tests/test_hunyuanvideo_load.py --sequence_model_parallel_size $num_gpus

torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503  sgl-diffusion/tests/test_llama_encoder.py


export SGL_DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503  sgl-diffusion/tests/test_clip_encoder.py
