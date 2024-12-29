## Run benchmark

### Benchmark SGLang with Radix Cache Offload 
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --tensor-parallel-size 4 --enable-hierarchical-cache

```
python3 bench_sglang.py --num_groups 100 --group_size 100 --context_length 1000 --cache_rate 0.8 
```

