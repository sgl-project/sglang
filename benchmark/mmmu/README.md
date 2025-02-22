## Run evaluation

### Evaluate sglang

```
python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl
```

It's recommended to reduce the memory usage by appending something ike `--mem-fraction-static 0.6` to the command above.

### Evaluate hf

```
python benchmark/mmmu/bench_hf.py --model-path Qwen/Qwen2-VL-7B-Instruct
```
