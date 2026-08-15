## Download data
```
git clone https://hf-mirror.com/datasets/google/boolq
```

## Convert parquet to json
```
bash parquet_to_json.sh
```
## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path ramblingpolymath/Qwen3-32B-W8A8 --port 30000
```

```
python3 bench_sglang.py
```
