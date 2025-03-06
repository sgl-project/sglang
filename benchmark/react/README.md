## Run benchmark

NOTE: This is an implementation for replaying a given trace for throughput/latency benchmark purposes. It is not an actual ReAct agent implementation.

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 100
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 100 --backend vllm
```


### Benchmark guidance
```
python3 bench_other.py --num-questions 100 --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

### Benchmark lmql

```
python3 bench_other.py --num-questions 100 --backend lmql --parallel 1
```
