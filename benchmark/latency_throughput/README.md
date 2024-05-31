### Download data
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
Install [FlashInfer](https://github.com/flashinfer-ai/flashinfer) if you want it to be enabled.


### SGLang
```
# use native attention
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --tp 1 --port 30000
# use flashinfer attention: --enable-flashinfer
# disable RadixAttention: --disable-radix-cache
```

```
# run ShareGPT
python3 bench_throughput.py --backend srt --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10 --port 30000
```

```
# run synthetic
python3 synthetic_benchmark.py --backend srt --tokenizer meta-llama/Llama-2-7b-chat-hf --num-prompt 1000 --request-rate 100 --input-len 1024 --output-len 256 --port 30000
```


### vLLM
```
python3 -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel 1 --disable-log-requests --swap-space 16 --port 21000
```

```
# run ShareGPT
python3 bench_throughput.py --backend vllm --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10 --port 21000
```

```
# run synthetic
python3 synthetic_benchmark.py --backend vllm --tokenizer meta-llama/Llama-2-7b-chat-hf --num-prompt 1000 --request-rate 100 --input-len 1024 --output-len 256 --port 30000
```


### LightLLM
```
python -m lightllm.server.api_server --model_dir ~/model_weights/Llama-2-7b-chat-hf --max_total_token_num 15600 --tokenizer_mode auto --port 22000
```

```
python3 bench_throughput.py --backend lightllm --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10 --port 22000
```
