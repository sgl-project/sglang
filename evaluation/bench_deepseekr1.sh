model=/data/pretrained-models/DeepSeek-R1-MXFP4-Preview

input_tokens=3584
output_tokens=1024
max_concurrency=64
num_prompts=128

# for profiling
#export SGLANG_TORCH_PROFILER_DIR=./
#export SGLANG_PROFILE_WITH_STACK=1

python3 -m sglang.bench_serving \
    --host localhost \
    --port 9000 \
    --model ${model} \
    --dataset-name random \
    --random-input ${input_tokens} \
    --random-output ${output_tokens} \
    --random-range-ratio 1.0 \
    --max-concurrency ${max_concurrency} \
    --num-prompt ${num_prompts} \
    2>&1 | tee log.client.log
