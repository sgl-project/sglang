## SRT Unit Tests

### Latency Alignment
Make sure your changes do not slow down the following benchmarks
```
# single gpu
python -m sglang.bench_latency --model-path meta-llama/Llama-2-7b-chat-hf --mem-fraction-static 0.8 --batch 32 --input-len 512 --output-len 256
python -m sglang.bench_latency --model-path meta-llama/Llama-2-7b-chat-hf --mem-fraction-static 0.8 --batch 1 --input-len 512 --output-len 256

# multiple gpu
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-70B --tp 8 --mem-fraction-static 0.6 --batch 32 --input-len 8192 --output-len 1
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-70B --tp 8 --mem-fraction-static 0.6 --batch 1 --input-len 8100 --output-len 32

# moe model
python -m sglang.bench_latency --model-path databricks/dbrx-base --tp 8 --mem-fraction-static 0.6 --batch 4 --input-len 1024 --output-len 32
```

### High-level API

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
cd test/lang
python3 test_srt_backend.py
```

### Performance

#### MMLU
```
cd benchmark/mmlu
```
Follow README.md to download the data.

```
python3 bench_sglang.py --nsub 3

# Expected performance on A10G
# Total latency: 8.200
# Average accuracy: 0.413
```

#### GSM-8K
```
cd benchmark/gsm8k
```
Follow README.md to download the data.

```
python3 bench_sglang.py --num-q 200

# Expected performance on A10G
# Latency: 32.103
# Accuracy: 0.250
```

#### More
Please also test `benchmark/hellaswag`, `benchmark/latency_throughput`.

### More Models

#### LLaVA

```
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
```

```
cd benchmark/llava_bench
python3 bench_sglang.py

# Expected performance on A10G
# Latency: 50.031
```

## SGLang Unit Tests
```
export ANTHROPIC_API_KEY=
export OPENAI_API_KEY=
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
cd test/lang
python3 run_all.py
```

## OpenAI API server
```
cd test/srt
python test_openai_server.py
```