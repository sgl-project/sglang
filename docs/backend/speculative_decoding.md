# Speculative Decoding

SGLang now provides an EAGLE-based speculative decoding option. The implementation aims to maximize speed and efficiency and is considered to be among the fastest in open-source LLM engines.

**Note:** Currently, Speculative Decoding in SGLang does not support radix cache.

### Performance Highlights

- Official EAGLE code ([SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)): ~200 tokens/s
- Standard SGLang Decoding: ~156 tokens/s
- EAGLE Decoding in SGLang: ~297 tokens/s
- EAGLE Decoding in SGLang (w/ `torch.compile`): ~316 tokens/s

All benchmarks below were run on a single H100.

## EAGLE Decoding

To enable EAGLE-based speculative decoding, specify the draft model (`--speculative-draft`) and the relevant EAGLE parameters:

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64
```

### EAGLE Decoding with `torch.compile`

You can also enable `torch.compile` for further optimizations and optionally set `--cuda-graph-max-bs`:

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE \
    --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 \
        --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --mem-fraction 0.6 \
            --enable-torch-compile --cuda-graph-max-bs 2
```