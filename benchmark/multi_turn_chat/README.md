### Benchmark sglang

Run Llama-7B

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

Run Mixtral-8x7B
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

Run Llama-7B

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Run Mixtral-8x7B

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model mistralai/Mixtral-8x7B-Instruct-v0.1 --disable-log-requests --port 21000 --tensor-parallel-size 8
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

Benchmark Llama-7B (short output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

Benchmark Llama-7B (long output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf --long
```
