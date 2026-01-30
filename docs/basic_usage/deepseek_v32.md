# DeepSeek V3.2 Usage

DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism powered by a lightning indexer, DeepSeek-V3.2 achieves efficiency improvements in long-context scenarios.

For reporting issues or tracking upcoming features, please refer to this [Roadmap](https://github.com/sgl-project/sglang/issues/11060).

Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) or [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) is the same as DeepSeek-V3.2-Exp except for the tool call parser.


## Installation

### Docker

```bash
# H200/B200
docker pull lmsysorg/sglang:latest

# MI350/MI355
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi35x

# MI300
# v0.5.8-rocm700-mi30x does not include PR #17504. Prefer the newest MI30x ROCm
# image tag from Docker Hub when available, or build from source (below).
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi30x


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
pip3 install -e "python"
```
## Launch DeepSeek V3.2 with SGLang

To serve [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) on 8xH200/B200 GPUs:

```bash
# Launch with TP + DP (Recommended)
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

# Launch with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention

# Launch with Pure TP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8

# Launch with TP on MI30x/MI35x
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --nsa-prefill-backend tilelang --nsa-decode-backend tilelang
```

### Configuration Tips
- **DP Attention (Recommended)**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`, so DP attention (`--dp 8 --enable-dp-attention`) is the recommended configuration for better stability and performance. All test cases use this configuration by default.
- **Pure TP Mode**: Launching with pure TP (without `--dp` and `--enable-dp-attention`) is also supported. Note that this mode has not been fully validated in PD disaggregation scenarios.
- **Short-sequence MHA prefill (adaptive)**: For short prefill sequences (default threshold: **2048 tokens**), the NSA backend uses standard MHA automatically (no extra flags). On H200 (SM90) this path uses the FlashAttention variable-length kernel; on B200 (SM100) it uses TRT-LLM ragged MHA. MHA uses `MHA_ONE_SHOT` for best performance. `MHA_ONE_SHOT` computes multi-head attention over all tokens (both cached prefix and newly extended tokens) in a single kernel invocation, avoiding the overhead of chunked KV cache processing. This achieves optimal throughput for short sequences where total sequence length fits within the chunk capacity limit.
- **Choices of Attention Kernels**: The attention backend is automatically set to `nsa` attention backend for DeepSeek V3.2 model. In this backend, different kernels for sparse prefilling/decoding are implemented, which can be specified by `--nsa-prefill-backend` and `--nsa-decode-backend` server arguments. The choices of nsa prefill/decode attention kernels include:
  - `flashmla_sparse`: `flash_mla_sparse_fwd` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, kv inputs.
  - `flashmla_kv`: `flash_mla_with_kvcache` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, fp8 k_cache inputs.
  - `fa3`: `flash_attn_with_kvcache` kernel from `flash_attn` library. Can only run on Hopper GPUs. It requires bf16 q, kv inputs.
  - `tilelang`: `tilelang` implementation that can run on GPU, HPU and NPU.
  - `aiter`: Aiter kernel on AMD HPUs. Can only be used as decode kernel.
- On the basis of performance benchmarks, the default configuration on H200 and B200 are set as follows :
  - H200: `flashmla_sparse` prefill attention (short-seq prefill uses MHA via FlashAttention varlen), `fa3` decode attention, `bf16` kv cache dtype.
  - B200: `flashmla_auto` prefill attention (short-seq prefill uses MHA via TRT-LLM ragged), `flashmla_kv` decode attention, `fp8_e4m3` kv cache dtype. `flashmla_auto` enables automatic selection of either `flashmla_sparse` or `flashmla_kv` kernel for prefill based on KV cache dtype, hardware, and heuristics. When FP8 KV cache is enabled and `total_kv_tokens < total_q_tokens * 512`, it uses the `flashmla_sparse` kernel; otherwise, it falls back to the `flashmla_kv` kernel. The heuristics may need to be tuned if the performance of either the `flashmla_sparse` or `flashmla_kv` kernel changes significantly.

## Multi-token Prediction
SGLang implements Multi-Token Prediction (MTP) for DeepSeek V3.2 based on [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved significantly on small batch sizes. Please look at [this PR](https://github.com/sgl-project/sglang/pull/11652) for more information.

Example usage with DP Attention:
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Example usage with Pure TP:
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

- The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
- The default value of  `--max-running-requests` is set to `48` for MTP. For larger batch sizes, this value should be increased beyond the default value.

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```


## Function Calling and Reasoning Parser
The usage of function calling and reasoning parser is the same as DeepSeek V3.1. Please refer to [Reasoning Parser](https://docs.sglang.io/advanced_features/separate_reasoning.html) and [Tool Parser](https://docs.sglang.io/advanced_features/tool_parser.html) documents.

To launch `DeepSeek-V3.2-Exp` with function calling and reasoning parser:
> Note: It is recommended to specify the chat-template, ensuring that you are within the sglang's root directory.
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv31 \
  --reasoning-parser deepseek-v3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja
```

To launch `DeepSeek-V3.2` with function calling and reasoning parser:
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv32 \
  --reasoning-parser deepseek-v3
```

`DeepSeek-V3.2-Speciale` doesn't support tool calling, so can only be launched with reasoning parser:
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Speciale \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --reasoning-parser deepseek-v3
```

## NVFP4 Checkpoint

To launch deepseek v3.2 [NVFP4 checkpoint](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) on Blackwell devices, the user needs to specify the quantization method as `modelopt_fp4`, and moe runner backend as one of `flashinfer_trtllm`(recommended), `flashinfer_cutlass` and `flashinfer_cutedsl`. Any other usage (parallelism, reasoning parser, ...) is the same as FP8 checkpoint.

An example launching command can be:
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32  --reasoning-parser deepseek-v3
```

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

To test long-context accuracy, run gsm8k with `--num-shots 20`. The results are very close to the 8 shots results:
```
Accuracy: 0.956
Invalid: 0.000
Latency: 29.545 s
Output throughput: 4418.617 token/s
```


### Accuracy Test with `gpqa-diamond`

Accuracy benchmark on long context can be tested on GPQA-diamond dataset with long output tokens and thinking enabled:
```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --thinking-mode deepseek-v3
```

The mean accuracy over 8 runs shows 0.797, which matches the number 0.799 in official tech report.
```bash
Repeat: 8, mean: 0.797
Scores: ['0.808', '0.798', '0.808', '0.798', '0.783', '0.788', '0.803', '0.793']
```

For Deepseek V3.2, Deepseek recommends setting the sampling parameters to temperature = 1.0, top_p = 0.95:

```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --top-p 0.95 --temperature 1.0 --thinking-mode deepseek-v3

Repeat: 8, mean: 0.840
Scores: ['0.848', '0.808', '0.848', '0.838', '0.879', '0.813', '0.838', '0.848']
```
which matches the official score, 0.824, as reported in the [Deepseek-V3.2 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf).

### Accuracy Test with `aime 2025`

Prepare the environment by installing NeMo-Skills in the docker or your own virtual environment:

  ```
  pip install git+https://github.com/NVIDIA/NeMo-Skills.git --ignore-installed blinker
  ```

Then launch the SGLang server:
```
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

**For `DeepSeek-V3.2` and `DeepSeek-V3.2-Speciale`**:

```
python3 -m sglang.launch_server   --model-path deepseek-ai/DeepSeek-V3.2   --trust-remote-code   --tp-size 8 --dp-size 8 --enable-dp-attention   --tool-call-parser deepseekv32   --reasoning-parser deepseek-v3
```

Run the following script to evaluate AIME 2025:
```
#! /bin/bash
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

ns prepare_data aime25

PORT=30000
BACKEND=sglang
MODEL="deepseek-ai/DeepSeek-V3.2-Exp" # Should be changed to the model name
MODEL_NAME="dsv32-fp8"

echo "Starting AIME25 evaluation with model $MODEL on port $PORT using backend $BACKEND..."
ns eval \
  --benchmarks=aime25:4 \
  --server_type=$BACKEND \
  --model=$MODEL \
  --server_address=http://localhost:${PORT}/v1 \
  --output_dir=nemo_skills_aime25_${MODEL_NAME}_output_${BACKEND}_$(date +%Y%m%d_%H%M%S) \
  ++chat_template_kwargs.thinking=true \
  ++inference.temperature=1.0 \
  ++inference.top_p=0.95 \
  ++inference.tokens_to_generate=64000
  # ++inference.tokens_to_generate=120000 for Speciale model
```

Test results (8*B200):

DeepSeek-V3.2-Exp：

| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 15040      | 1673        | 87.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 15040      | 1673        | 90.00%                | 0.00%     |
| pass@4             | 30          | 15040      | 1673        | 90.00%                | 0.00%     |


DeepSeek-V3.2:
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 13550      | 1632        | 92.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 13550      | 1632        | 94.71%                | 0.00%     |
| pass@4             | 30          | 13550      | 1632        | 96.67%                | 0.00%     |


DeepSeek-V3.2-Speciale:
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 24155      | 3583        | 95.00% ± 1.92%        | 0.00%     |
| majority@4         | 30          | 24155      | 3583        | 95.83%                | 0.00%     |
| pass@4             | 30          | 24155      | 3583        | 100.00%               | 0.00%     |



## DSA long sequence context parallel optimization(experimental)

**Note: This feature is only verified on Hopper machines**

For context parallel in DeepSeek V3.2 model, we provide two different modes of splitting tokens, which can be controlled with argument `--nsa-prefill-cp-mode`.

### In sequence splitting (default setting)

The first mode can be enabled by `--nsa-prefill-cp-mode in-seq-split`. This mode implements context parallel for DSA by splitting the sequence uniformly between context parallel ranks. At attention stage, each cp rank computes the indexer results of sharded sequence, and collects the whole kv cache through all gather operator.

The communication group for context parallel reuses the one for attention tp, thus `cp_size` equals `atten_tp_size = tp_size / dp_size`.

Note that in sequence splitting mode has the following restrictions:
- The batch size is restricted to 1 for prefill batches
- Multi-node/PD disaggregation is still not supported
- `moe_dense_tp_size=1`, `kv_cache_dtype = "bf16"`, `moe_a2a_backend = "deepep"`
- To ensure `cp_size > 1`, the passed in `tp_size` must be larger than `dp_size`

For more details, please refer to PR https://github.com/sgl-project/sglang/pull/12065.

Example:
```bash
# In-seq splitting mode launched with EP + DP
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --ep 8 --dp 2 --enable-dp-attention --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode in-seq-split --max-running-requests 32
```

### Round robin splitting

This mode can be enabled by specifying the parameter `--nsa-prefill-cp-mode round-robin-split`, which distributes tokens across ranks based on `token_idx % cp_size`.

In this scenario, compared with the aforementioned method, it additionally supports the fused MoE backend (the fused MoE backend may deliver better performance than DeepEP in single-machine scenarios), FP8 KV-cache, and multi-batch prefill inference. But it cannot be enabled with dp attention together.

For more details, please refer to PR https://github.com/sgl-project/sglang/pull/13959.

Example usage:
```bash
# Launch with FusedMoe + CP8
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split --max-running-requests 32
```
### Pipeline Parallel + Context Parallel (PP + CP)

This mode combines Pipeline Parallelism (PP) and Context Parallelism (CP) to scale across multiple nodes, which can achieve better throughput and Time To First Token (TTFT). Note that this method has only been tested on H20 96G.

#### Standard Usage

To launch with PP=2 and CP (via `round-robin-split` mode) on 2 nodes. This configuration uses the fused MoE kernel by default, which generally provides better performance.

For related development details, please refer to:
- Fused MoE + CP support: [PR #13959](https://github.com/sgl-project/sglang/pull/13959)
- PP + CP support: [Issue #15358](https://github.com/sgl-project/sglang/issues/15358) and [PR #16380](https://github.com/sgl-project/sglang/pull/16380)

Node 0:
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

Node 1:
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

#### PD Disaggregation with PP + CP

If using PD (Prefill-Decode) Disaggregation, the Prefill nodes can be configured with PP + CP as follows.

Prefill Node 0:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32 \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

Prefill Node 1:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32-prefill \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

For the Decode nodes, it is recommended to use the **EP mode**.
