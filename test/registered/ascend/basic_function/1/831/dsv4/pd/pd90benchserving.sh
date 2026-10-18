unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export WP=/home/weights/DeepSeek-V4-Pro-0813-w4a8

export SGLANG_TORCH_PROFILER_DIR=/home/y30082119/sglang_dsv4_pro/prof

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host http://192.168.25.209 \
  --port 6699 \
  --dataset-name generated-shared-prefix \
  --backend sglang \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 117965 \
  --gsp-question-len 13107 \
  --gsp-output-len 1024 \
  --max-concurrency 8 \
  --random-range-ratio 1 \
  --seed 100 \
  --model $WP \
  --tokenizer $WP \
  --warmup-requests 16
