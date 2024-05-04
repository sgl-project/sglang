## Download the dataset

```
wget -O agent_calls.jsonl https://drive.google.com/uc?export=download&id=19qLpD45e9JGTKF2cUjJJegwzSUEZEKht
```

## Run benchmark

Ensure that this benchmark is run in a serial manner (using --parallel 1) to preserve any potential dependencies between requests.

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-events 1000 --parallel 1
```

### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-events 1000 --backend vllm --parallel 1
```

### Benchmark guidance
```
python3 bench_other.py --num-events 1000 --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

### Benchmark lmql

```
python3 bench_other.py --num-events 1000 --backend lmql --parallel 1
```
