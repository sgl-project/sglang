## Run evaluation

### Evaluate sglang

Host the VLM:

```
python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000
```

Benchmark:

```
python benchmark/mmmu/bench_sglang.py --port 30000
```

It's recommended to reduce the memory usage by appending something ike `--mem-fraction-static 0.6` to the command above.

### Evaluate hf

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```

Some popular model results:

1. Qwen/Qwen2-VL-2B-Instruct: 0.241
2. Qwen/Qwen2-VL-7B-Instruct: 0.255
3. Qwen/Qwen2.5-VL-3B-Instruct: 0.245
4. Qwen/Qwen2.5-VL-7B-Instruct: 0.242
