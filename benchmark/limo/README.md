## Run benchmark

This benchmark is primarily intended to be used with reasoning models like `DeepSeek-R1-Distill-Qwen-1.5B`.

### Benchmark sglang

Launch server

```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000
```

Note that depending on the GPU this benchmark will take quiet some time. To employ data parallelism please use:

```bash
python3 -m sglang_router.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000 --dp-size 4
```

Benchmark

```bash
python3 bench_sglang.py --parallel 256
```