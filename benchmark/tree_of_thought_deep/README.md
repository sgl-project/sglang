## Download data
```
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

## Run benchmark

NOTE: This is an implementation for throughput/latency benchmark purposes. The prompts are not tuned to achieve good accuracy on the GSM-8K tasks.

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 32
python3 bench_sglang.py --num-questions 16 --parallel 1
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 32 --backend vllm
```


### Benchmark lightllm
```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000
```

```
python3 bench_other.py --num-questions 32 --backend lightllm
```


### Benchmark guidance
```
python3 bench_other.py --num-questions 8 --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

### Benchmark lmql

```
python3 bench_other.py --num-questions 8 --backend lmql --parallel 1
```
