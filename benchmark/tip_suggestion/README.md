## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 64
python3 bench_sglang.py --num-questions 32 --parallel 1
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --backend vllm --num-questions 64
```


### Benchmark guidance
```
python3 bench_other.py --backend guidance --num-questions 32 --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

### Benchmark lmql

```
python3 bench_other.py --backend lmql --num-questions 32 --parallel 1
```
