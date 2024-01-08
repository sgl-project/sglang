### Download data
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Performance

- Model: Llama-2-7b-chat-hf
- `--num-prompts 2000 --request-rate 200`
- On 4 A10 (24G) GPUs

| Backend     | Throughput      | Latency  |
| ----------- | --------------- | -------- |
| srt         | 5.82 requests/s | 343.54 s |
| vllm==0.2.6 | 3.93 requests/s | 509.08 s |
| vllm==0.2.7 | 5.02 requests/s | 398.25 s |

 
### SGLang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_throughput.py --backend srt --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10 --port 30000
```


### vLLM
```
python3 -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --swap-space 16
```

```
python3 bench_throughput.py --backend vllm --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10
```


### LightLLM
```
python -m lightllm.server.api_server --model_dir ~/model_weights/Llama-2-7b-chat-hf --max_total_token_num 15600 --tokenizer_mode auto --port 22000
```

```
python3 bench_throughput.py --backend lightllm --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --request-rate 10 --port 22000
```
