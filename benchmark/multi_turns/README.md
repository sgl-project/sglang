### Benchmark sglang

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf
```

```
python bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf --long
```

### Benchmark vLLM

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm
```

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm --long
```

### Benchmark guidance

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1
```

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --long
```