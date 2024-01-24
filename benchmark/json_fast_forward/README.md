## Run benchmark

### Dependencies

```
llama_cpp_python          0.2.32
guidance                  0.1.10
vllm                      0.2.7
outlines                  0.0.24
```

### Benchmark sglang

Run Llama-7B

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 
```

Benchmark

```
python3 bench_sglang.py
```

### Benchmark vllm

Run Llama-7B

```
python3 -m outlines.serve.serve --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Benchmark

```
python3 bench_other.py --backend vllm
```

### Benchmark guidance (seems not supported)

Run Llama-7B and benchmark

```
python3 bench_other.py --backend guidance --parallel 1
```
