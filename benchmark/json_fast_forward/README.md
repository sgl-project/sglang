## Run benchmark

### Dependencies

```
llama_cpp_python          0.2.38
guidance                  0.1.10
vllm                      0.2.7
outlines                  0.0.25
```

### Build dataset

When benchmarking long document information retrieval, run the following command to build the dataset:

```bash
pip install wikipedia
python3 build_dataset.py
```

### Benchmark sglang

Run Llama-7B

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 
```

Benchmark Character Generation

```bash
python3 bench_sglang.py --mode character
```

Benchmark City Information Retrieval

```bash
python3 bench_sglang.py --mode city
```


### Benchmark vllm

Run Llama-7B

```bash
python3 -m outlines.serve.serve --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Benchmark Character Generation

```bash
python3 bench_other.py --mode character --backend vllm
```

Benchmark City Information Retrieval

```bash
python3 bench_other.py --mode city --backend vllm
```

### Benchmark guidance

Run Llama-7B and benchmark character generation

```bash
python3 bench_other.py --mode character --backend guidance --parallel 1
```

Run Llama-7B and benchmark city information retrieval

```bash
python3 bench_other.py --mode city --backend guidance --parallel 1
```
