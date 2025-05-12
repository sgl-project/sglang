## Download data
```
bash download_data.sh
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --nsub 10
```

```
# OpenAI models
python3 bench_sglang.py --backend gpt-3.5-turbo --parallel 8
```

### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --nsub 10 --backend vllm
```


### Benchmark lightllm
```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000

# V100
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 4500 --port 22000
```

```
python3 bench_other.py --nsub 10 --backend lightllm
```


### Benchmark guidance
```
python3 bench_other.py --nsub 10 --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```


### Benchmark lmql
```
CUDA_VISIBLE_DEVICES=0,1 lmql serve-model meta-llama/Llama-2-7b-chat-hf --cuda --port 23000
```

```
python3 bench_other.py --nsub 10 --backend lmql --parallel 2
```
