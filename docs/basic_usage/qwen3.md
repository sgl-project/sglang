# Qwen3-Next Usage

SGLang has supported Qwen3-Next-80B-A3B-Instruct and Qwen3-Next-80B-A3B-Thinking since [this PR](https://github.com/sgl-project/sglang/pull/10233).

## Launch Qwen3-Next with SGLang

To serve Qwen3-Next models on 4xH100/H200 GPUs:

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### Configuration Tips
- `--max-mamba-cache-size`: Adjust `--max-mamba-cache-size` to increase mamba cache space and max running requests capability. It will decrease KV cache space as a trade-off. You can adjust it according to workload.
- `--mamba-ssm-dtype`: `bfloat16` or `float32`, use `bfloat16` to save mamba cache size and `float32` to get more accurate results. The default setting is `float32`.
- `--mamba-full-memory-ratio`: The ratio of mamba state memory to full kv cache memory. The default is 0.9.

### Mamba Radix Cache
SGLang supports prefix caching for Qwen3-Next models named `MambaRadixCache`, which improves inference speed by reusing computation results. There are two versions of `MambaRadixCache`:
- `no_buffer`: The default version, which is also other hybrid linear models' choice. When it is enabled, SGLang will automatically close overlap schedule for compatibility reasons.
- `extra_buffer`: An optimized version that is compatible with features like page size > 1, overlap schedule, and speculative decoding. It also supports storing mamba state in branching positions. However, it requires two extra mamba spaces for a ping-pong buffer for each request. To enable it, add the argument `--mamba-scheduler-strategy extra_buffer` when launching the server.

### EAGLE Speculative Decoding
**Description**: SGLang has supported Qwen3-Next models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**Usage**:
Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

Details can be seen in [this PR](https://github.com/sgl-project/sglang/pull/10233).
