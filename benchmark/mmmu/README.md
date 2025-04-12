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

Benchmark Results:

| Model                   | SGLang | HuggingFace |
|-------------------------|--------|-------------|
| Qwen2-VL-7B-Instruct   | 0.476  | -            |
| Qwen2.5-VL-7B-Instruct | 0.477  | 0.504        |
| MiniCPM-V-2.6          | 0.435  | —            |
| MiniCPM-O-2_6          | 0.401 | - |
| Deepseek-Janus-Pro-7B  | 0.373  | -            |
| Deepseek-VL2           | 0.405 | - |
| Gemma-3-it-4B          | 0.41   | 0.403        |
| llama3-llava-next-8b | 0.245 | - |
| llava-v1.6-mistral-7b-sglang | 0.338 | - |
| llava-onevision-qwen2-7b-ov | 0.423 | - |
| Mlama - Llama-3.2-11B-Vision-Instruct | 0.321 | - |
