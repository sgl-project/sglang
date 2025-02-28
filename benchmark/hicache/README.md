## Run synthetic multi-turn benchmark

```
# SGLang server with radix cache disabled
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --disable-radix-cache

# SGLang server with radix cache on and first-come-first-serve policy
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --schedule-policy fcfs

# The default SGLang server with radix cache on and long-prefix-match policy
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000

# SGLang server with hierarchical radix cache enabled
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --enable-hierarchical-cache

```

```
python bench_multiturn.py --model-path Qwen/Qwen2.5-14B-Instruct
```

Note: The performance gain of hierarchical caching depends on the ratio of reusable tokens to GPU memory capacity. The more tokens to be reused, the larger the model, and the more constrained the GPU memory size, the greater the benefit one can expect from hierarchical caching.


## More benchmarks to be added
