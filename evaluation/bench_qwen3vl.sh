model=/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/

input_tokens=1000
output_tokens=2000
num_prompts=128
max_concurrency=64
image_count=1
image_resolution=800x800

# for profiling
#export SGLANG_TORCH_PROFILER_DIR=./profile_log
#add --profile as argument

echo "bench model: ${model}"
echo "input tokens: ${input_tokens}"
echo "output tokens: ${output_tokens}"
echo "image-count: ${image_count}"
echo "image-resolution: ${image_resolution}"
echo "max concurrency: ${max_concurrency}"
echo "num prompts: ${num_prompts}"

python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --port 9000 \
    --dataset-name image \
    --num-prompts ${num_prompts} \
    --image-count ${image_count} \
    --image-resolution ${image_resolution} \
    --random-input-len ${input_tokens} \
    --random-output-len ${output_tokens} \
    --max-concurrency ${max_concurrency} \
    2>&1 | tee log.client.log
