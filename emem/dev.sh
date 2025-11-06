for _ in {1..2}; do
  ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
  ps aux | grep "sglang::" | grep -v grep | awk '{print $2}' | xargs kill -9
  sleep 1
done

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
rm -rf nohup.out
nohup python3 -m sglang.launch_server \
  --log-level debug \
  --model /data/models/tiny-random-llama-4-8E \
  --attention-backend fa3 \
  --mem-fraction-static 0.3 \
  --hybrid-kvcache-ratio 1.0 \
  --context-length 32767 &

sleep 3

tail -f nohup.out

python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 8192 --random-input 24576 --random-output 1024 --random-range-ratio 0.5 \
  --max-concurrency 2048

curl -L -X POST 'http://127.0.0.1:30000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
--data-raw '{
  "messages": [
    {
      "content": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
      "role": "user"
    }
  ],
  "model": "xxx",
  "max_tokens": 64,
  "stream": false
}'
