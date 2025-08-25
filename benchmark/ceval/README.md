## Download data
```
git lfs clone https://huggingface.co/datasets/ceval/ceval-exam
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path ramblingpolymath/Qwen3-32B-W8A8 --port 30000
```

```
python3 bench_sglang.py
```
