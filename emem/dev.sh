ps aux | grep 'sglang.launch_server' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
ps aux | grep 'sglang::scheduler' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
ps aux | grep 'sglang::detokenizer' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
rm -rf nohup.out
nohup python3 -m sglang.launch_server \
  --log-level debug --disable-cuda-graph \
  --model /data/models/tiny-random-llama-4-8E \
  --attention-backend fa3 \
  --mem-fraction-static 0.3 \
  --hybrid-kvcache-ratio 1.0 \
  --context-length 32767 &

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
      "content": "Hi!",
      "role": "user"
    }
  ],
  "model": "xxx",
  "max_tokens": 64,
  "stream": false
}'
