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

You can use `--response-answer-regex` to specify how to extract the answer from the response string. E.g.,
```
python3 -m sglang.launch_server --model-path zai-org/GLM-4.1V-9B-Thinking --reasoning-parser glm45

python3 bench_sglang.py --response-answer-regex "<\|begin_of_box\|>(.*)<\|end_of_box\|>" --concurrency 64
```

You can use `--extra-request-body` to specify additional OpenAI request parameters. E.g.,
```
python3 bench_sglang.py --extra-request-body '{"max_new_tokens": 128, "temperature": 0.01}'
```

### Evaluate HF

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```

# Profiling MMMU
You should use the standard instructions found in the [dedicated profiling doc](../../docs/developer_guide/benchmark_and_profiling.md) if running this benchmark with the profile option. We recommend using `--concurrency 1` for consistency, which makes profiling and debugging easier.
