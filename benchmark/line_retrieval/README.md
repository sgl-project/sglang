## Download data

```
wget https://raw.githubusercontent.com/merrymercy/merrymercy.github.io/master/files/random_words.json
python3 gen_data.py --number 1000
```

## Run benchmark

### Benchmark sglang
```
python3 -m sglang.launch_server --model-path codellama/CodeLlama-7b-hf --port 30000
```

```
python3 bench_sglang.py --src-index 600 --num-q 50 --parallel 1
```


###

```
# original
Accuracy: 0.940, latency: 332.83 s

# parallel encoding (no_adjust, offset = 1000)
Accuracy: 0.760, latency: 238.46 s

# parallel encoding (no_adjust, offset = 3000)
Accuracy: 0.760, latency: 238.46 s

# parallel encoding (no_adjust, offset = 0)
Accuracy: 0.520, latency: 238.46 s

# parallel encoding (adjust_cache)
Accuracy: 0.460, latency: 257.66 s
```
