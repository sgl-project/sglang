export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python -m sglang.launch_server --model-path /mnt/petrelfs/shaojie/code/ckpts/OpenGVLab/InternVL3-8B-hf --tp 1 --chat-template internvl-2-5 --log-requests --log-requests-level 2 --port 10010 --host 0.0.0.0

# python -m sglang.launch_server --model-path /mnt/petrelfs/shaojie/code/ckpts/OpenGVLab/InternVL3-8B --tp 1 --chat-template internvl-2-5 --log-requests --log-requests-level 2 --port 10010 --host 0.0.0.0