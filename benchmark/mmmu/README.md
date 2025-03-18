## Run evaluation

### Evaluate sglang

```
python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000
```

```
python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000
```

It's recommended to reduce the memory usage by appending something ike `--mem-fraction-static 0.6` to the command above.

### Evaluate hf

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```

Some popular model results:
1. Qwen/Qwen2-VL-7B-Instruct(sglang): 0.48
2. Qwen/Qwen2-VL-7B-Instruct(hf): 0.482
3. OpenGVLab/InternVL2_5-38B(sglang): 0.612
4. OpenGVLab/InternVL2_5-38B(hf): 0.61