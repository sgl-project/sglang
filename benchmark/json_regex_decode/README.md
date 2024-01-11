## Run benchmark

### Dependencies

```
llama_cpp_python          0.2.19
guidance                  0.1.10
vllm                      0.2.5
outlines                  0.0.22
```

### Benchmark sglang
```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 
```

```
python3 bench_sglang.py --num-questions 5
```


### Benchmark vllm
```
python3 -m outlines.serve.serve --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

```
python3 bench_other.py --backend vllm --num-questions 5
```


### Benchmark guidance
```
python3 bench_other.py --backend guidance --num-questions 5 --parallel 1
```


### Build dataset
```
pip install wikipedia
python3 build_dataset.py
```
