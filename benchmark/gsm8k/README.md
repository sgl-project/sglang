## Run benchmark

### Using GSM8K Platinum

GSM8K Platinum is a revised version of the GSM8K test set with corrected labels and removed ambiguous questions. It can be more stable than the original GSM8K dataset. It's a drop-in replacement that can be used by adding the `--platinum` flag:

```
python3 bench_sglang.py --num-shots 8 --num-questions 1209 --parallel 1209 --platinum
```

For more information, see: https://huggingface.co/datasets/madrylab/gsm8k-platinum

### Benchmark sglang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 200
```


### Benchmark vllm
```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 200 --backend vllm
```


### Benchmark lightllm
```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000
```

```
python3 bench_other.py --num-questions 200 --backend lightllm
```


### Benchmark guidance
```
python3 bench_other.py --num-questions 200 --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```


### Benchmark lmql
```
CUDA_VISIBLE_DEVICES=0,1 lmql serve-model meta-llama/Llama-2-7b-chat-hf --cuda --port 23000
```

```
python3 bench_other.py --num-questions 100 --backend lmql --parallel 2
```
