## Download data
```
bash download_data.sh
```

## Run benchmark

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --nsub 10
```

```
# OpenAI models
python3 bench_sglang.py --backend gpt-3.5-turbo --parallel 8
```
