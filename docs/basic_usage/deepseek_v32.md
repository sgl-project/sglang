# DeepSeek V3.2 Usage

[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism powered by a lightning indexer, DeepSeek-V3.2 achieves efficiency improvements in long-context scenarios.

For reporting issues or tracking upcoming features, please refer to this [Roadmap](https://github.com/sgl-project/sglang/issues/11060).

## Installation

### Docker

```bash
# H200/B200
docker pull lmsysorg/sglang:latest

# MI350/MI355
docker pull lmsysorg/sglang:dsv32-rocm

# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

### Build From Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install -e "python[all]"

# Install flash_mla
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```
## Launch DeepSeek V3.2 with SGLang

To serve DeepSeek-V3.2-Exp on 8xH200/B200 GPUs:

```bash
# Launch with TP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

# Launch with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention
```

### Configuration Tips
- **DP Attention**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`, so DP attention is enabled by default for better stability and performance. The feature of launching with pure TP is still under development.
- **Choices of Attention Kernels**: The attention backend is automatically set to `nsa` attention backend for DeepSeek V3.2 model. In this backend, different kernels for sparse prefilling/decoding are implemented, which can be specified by `--nsa-prefill-backend` and `--nsa-decode-backend` server arguments. The choices of nsa prefill/decode attention kernels include:
  - `flashmla_sparse`: `flash_mla_sparse_fwd` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs.
  - `flashmla_kv`: `flash_mla_with_kvcache` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs.
  - `fa3`: `flash_attn_with_kvcache` kernel from `flash_attn` library. Can only run on Hopper GPUs.
  - `tilelang`: `tilelang` implementation that can run on GPU, HPU and NPU.
  - `alter`: Alter kernel on AMD HPUs. Can only be used as decode kernel.
- On the basis of performance benchmarks, the default configuration on H200 and B200 are set as follows :
  - H200: `flashmla_sparse` prefill attention, `fa3` decode attention, `bf16` kv cache dtype.
  - B200: `flashmla_kv` prefill attention, `flashmla_kv` decode attention, `fp8_e4m3` kv cache dtype.
  - Currently we don't enable `prefill=flashmla_sparse` with `decode=flashmla_kv` due to latency caused by kv cache quantization operations. In the future we might shift to this setting after attention/quantization kernels are optimized.

## Multi-token Prediction
SGLang implements Multi-Token Prediction (MTP) for DeepSeek V3.2 based on [EAGLE speculative decoding](https://docs.sglang.ai/advanced_features/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved significantly on small batch sizes. Please look at [this PR](https://github.com/sgl-project/sglang/pull/11652) for more information.

Example usage:
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
- The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
- The default value of  `--max-running-requests` is set to `48` for MTP. For larger batch sizes, this value should be increased beyond the default value.


## Function Calling and Reasoning Parser
The usage of function calling and reasoning parser is the same as DeepSeek V3.1. Please refer to [Reasoning Parser](https://docs.sglang.ai/advanced_features/separate_reasoning.html) and [Tool Parser](https://docs.sglang.ai/advanced_features/tool_parser.html) documents.

## PD Disaggregation

Prefill Command:
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode prefill \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --disaggregation-bootstrap-port 8998 \
        --mem-fraction-static 0.9 \
```

Decode command:
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode decode \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --mem-fraction-static 0.9 \
```

Router command:
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill $PREFILL_ADDR 8998 \
  --decode $DECODE_ADDR \
  --host 127.0.0.1 \
  --port 8000 \
```

If you need more advanced deployment methods or production-ready deployment methods, such as RBG or LWS-based deployment, please refer to [references/multi_node_deployment/rbg_pd/deepseekv32_pd.md](../references/multi_node_deployment/rbg_pd/deepseekv32_pd.md). Additionally, you can also find startup commands for DeepEP-based EP parallelism in the aforementioned documentation.


## Benchmarking Results

### Accuracy Test with `gsm8k`
A simple accuracy benchmark can be tested with `gsm8k` dataset:
```bash
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

The result is 0.956, which matches our expectation:
```bash
Accuracy: 0.956
Invalid: 0.000
Latency: 25.109 s
Output throughput: 5226.235 token/s
```


### Accuracy Test with `gpqa-diamond`

Accuracy benchmark on long context can be tested on GPQA-diamond dataset with long output tokens and thinking enabled:
```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 120000 --repeat 8 --thinking-mode deepseek-v3
```

The mean accuracy over 8 runs shows 0.797, which matches the number 79.9 in official tech report.
```bash
Repeat: 8, mean: 0.797
Scores: ['0.808', '0.798', '0.808', '0.798', '0.783', '0.788', '0.803', '0.793']
```
