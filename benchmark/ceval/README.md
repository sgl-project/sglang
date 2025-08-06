## Download data
```
git lfs clone https://huggingface.co/datasets/ceval/ceval-exam
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py
```
