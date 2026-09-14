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
