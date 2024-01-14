### Benchmark sglang

Run llama-7b

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

Run mixtral-8x7b
(When there is a CUDA out-of-memory error, try to reduce the `--mem-fraction-static`)

```
python3 -m sglang.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --tp-size 8
```

Benchmark(short output)

```
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf
```

Benchmark(long output)

```
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf --long
```

### Benchmark vLLM

Run llama-7b

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Run mixtral-8x7b

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model mistralai/Mixtral-8x7B-Instruct-v0.1 --disable-log-requests --port 21000
```

Benchmark(short output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm
```

Benchmark(long output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm --long
```

### Benchmark guidance

Benchmark llama-7b(short output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1
```

Benchmark llama-7b(long output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --long
```