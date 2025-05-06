## Run evaluation

### Evaluate sglang

Host the VLM:

```
python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000
```

It's recommended to reduce the memory usage by appending something like `--mem-fraction-static 0.6` to the command above.

Benchmark:

```
python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16
```

You can adjust the `--concurrency` to control the number of concurrent OpenAI calls.

### Evaluate hf

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```
