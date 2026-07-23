## Run benchmark

### Build dataset
```
pip install wikipedia
python3 build_dataset.py
```

### Dependencies

```
llama_cpp_python          0.2.19
guidance                  0.1.10
vllm                      0.2.5
outlines                  0.0.22
```

### Benchmark sglang

Run Llama-7B

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

Run Mixtral-8x7B

```
python3 -m sglang.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --tp-size 8
```

Benchmark

```
python3 bench_sglang.py --num-questions 10
```


### Benchmark Outlines + vLLM

Run Llama-7B

```
python3 -m outlines.serve.serve --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Benchmark

```
python3 bench_other.py --backend outlines --num-questions 10
```


### Benchmark guidance

Run Llama-7B and benchmark

```
python3 bench_other.py --backend guidance --num-questions 10 --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```
