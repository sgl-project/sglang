# Qwen3.8-27B MeetUp results on Ascend

Xuuuhf · 2026-11-14

## Accuracy and serving performance

Both models ran on four Ascend 910B3 NPUs (64 GB each) with SGLang `0.0.0.dev22121+g0fa9ff55f.d20261112`. GSM8K accuracy uses the first 200 test questions under the same EvalScope settings.

The native SGLang benchmark sent 20 requests at concurrency 20. Each request had 128,000 input tokens, reused 115,200 cached tokens (90%), and generated 1,000 tokens. Each row is the highest-output-throughput run observed for that model among completed runs with 90% cache reuse and mean TPOT ≤ 50 ms.

| Model | GSM8K accuracy | Mean TPOT (ms) | P99 TPOT (ms) | Mean TTFT (s) | Output tokens/s | Requests with TPOT ≤ 50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B BF16 | 199/200 (99.5%) | 36.86 | 66.06 | 39.44 | 250.12 | 16/20 |
| Qwen3.8-27B W8A8 | 198/200 (99.0%) | 38.30 | 64.50 | 33.29 | 267.18 | 14/20 |

Both runs used TP=4, attention DP=2, a 131,072-token context, a 16,384-token prefill chunk (8,192 per DP rank), and a limit of 20 running requests. NEXTN used three steps and four draft tokens. The W8A8 run used ModelSlim quantization. Both DP ranks received the shared prefix before timing.

Across these two runs, W8A8 delivered 6.82% more output tokens per second and answered one fewer GSM8K question correctly.

## BF16 launch command

Increase the context limit from 16,384 to 131,072 so a request can fit 128,000 input and 1,000 output tokens. Enable the hybrid prefix cache with `extra_buffer`. Reduce the effective prefill chunk from 16,384 to 8,192 tokens per attention-DP rank.

Set `MODEL_ROOT` to the parent directory of `Qwen3.8-27B`. Run each service separately on the four NPUs.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export ASCEND_USE_FIA=1 STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=500 HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
sglang serve \
  --model-path "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the weights}/Qwen3.8-27B" \
  --host 127.0.0.1 \
  --port 6688 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --dp-size 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --context-length 131072 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.85 \
  --max-running-requests 20 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --enable-multimodal \
  --mm-attention-backend ascend_attn \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-bs-decode 1 2 4 6 8 10 12 16 20 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

## W8A8 launch command

Use ModelSlim W8A8 weights and reduce the server’s maximum running requests from 64 to 20 to match the measured concurrency. The selected configuration uses an effective prefill chunk of 8,192 tokens per rank and NEXTN with three steps and four draft tokens.

Set `MODEL_ROOT` to the parent directory of `Qwen3.8-27B-w8a8`. Run each service separately on the four NPUs.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export ASCEND_USE_FIA=1 STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=500 HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
sglang serve \
  --model-path "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the weights}/Qwen3.8-27B-w8a8" \
  --host 127.0.0.1 \
  --port 6688 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --dp-size 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --context-length 131072 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.85 \
  --max-running-requests 20 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --enable-multimodal \
  --mm-attention-backend ascend_attn \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-bs-decode 1 2 4 6 8 10 12 16 20 \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --quantization modelslim
```
