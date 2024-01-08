## SRT Unit Tests

### Low-level API
```
cd sglang/test/srt/model

python3 test_llama_low_api.py
python3 test_llama_extend.py
python3 test_llava_low_api.py
python3 bench_llama_low_api.py
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

### More Models

#### LLaVA

```
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
```

```
cd benchmark/llava_bench
python3 bench_sglang.py
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
