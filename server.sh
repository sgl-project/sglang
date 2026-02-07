export CUDA_VISIBLE_DEVICES=0,3,4,5
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo,+bytecode,+graph_breaks,+recompiles,+aot"

python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 4 \
    --load-format dummy \
    --enable-piecewise-cuda-graph