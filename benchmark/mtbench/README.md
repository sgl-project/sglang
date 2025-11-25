## Download Dataset

```sh
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 80
```

### Benchmark sglang EAGLE
```
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct --speculative-algo EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --dtype float16 --port 30000
```

```
python3 bench_sglang_eagle.py --num-questions 80 --parallel 1
```

### Benchmark sglang EAGLE3 with dynamic speculative decoding
```
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --watchdog-timeout 300 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path  QiaolingChen/EAGLE3-DeepSeek-R1-Distill-Qwen-14B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 32 \
    --enable-dynamic-spec \
    --speculative-batch-size-threshold 16 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.9 \
    --max-total-tokens 65536 \
    --max-prefill-tokens 4096
```

```
python3 bench_sglang_adaserve.py
```

### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 80 --backend vllm
```


### Benchmark lightllm
```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000
```

```
python3 bench_other.py --num-questions 80 --backend lightllm
```
