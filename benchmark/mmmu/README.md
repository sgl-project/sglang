## Run evaluation

### Evaluate sglang

Host the VLM:

```
python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000
```

It's recommended to reduce the memory usage by appending something like `--mem-fraction-static 0.6` to the command above.

Benchmark:

```
python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16
```

You can adjust the `--concurrency` to control the number of concurrent OpenAI calls.

You can use `--lora-path` to specify the LoRA adapter to apply during benchmarking. E.g.,
```
# Launch server with LoRA enabled
python -m sglang.launch_server --model-path microsoft/Phi-4-multimodal-instruct --port 30000 --trust-remote-code --disable-radix-cache --lora-paths vision=<LoRA path>

# Apply LoRA adapter during inferencing
python -m benchmark/mmmu/bench_sglang.py --concurrency 8 --lora-path vision
```

### Evaluate hf

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```
