## Motivation

Record the Qwen3.8-27B MeetUp deployment, GSM8K evaluation, and long-context prefix-cache tuning workflow on four Ascend 910B3 devices. This provides a BF16-versus-W8A8 experiment record for the dedicated MeetUp branch.

MeetUp task: Qwen3.8-27B deployment and tuning. Author / developer ID: **66887-hsk**.

## Modifications

- Add `docs/sglang-practice-for-meetup/66887-hsk-Qwen3.8-27B-report.md` with complete environment variables, launch commands, evaluation commands, prefix-only warmup and benchmark scripts, all recorded measurements, and documentation feedback.
- Preserve the original commands, screenshots, terminal records, per-example predictions, and benchmark JSONL files in the evidence package.
- State the binary token-length convention, measured cache hit rate, request budget, and output throughput explicitly.
- Separate final measurements from screening results, and report Mean TPOT, P99 TPOT, and the per-request TPOT pass rate.
- Target `ifmn/npu/sglang-practice-for-meetup`. The optional general performance-documentation patch is prepared separately for `main`.

## Accuracy Tests

GSM8K, 200 examples, the default 4-shot prompt, seed 42, temperature 0, and evaluation concurrency 8. The model's default thinking mode is retained, with no additional output-token cap. Both final evaluations completed 200 samples with zero API errors.

| Model | Non-MTP baseline | MTP final |
| --- | --- | --- |
| Qwen3.8-27B | Not measured at 200 examples | 197/200 (98.50%) |
| Qwen3.8-27B-w8a8 | 195/200 (97.50%) | 195/200 (97.50%) |

The input-message contents were checked for equivalence across all 200 final samples. Both final servers use the same tokenizer path. Predicted answers, review records, and score totals were cross-checked. These fixed-sample results do not establish a general accuracy ranking. Earlier 100-example stage results are retained separately in the report.

## Speed Tests and Profiling

This experiment uses binary K: **131,072 input token IDs = 117,965 shared-prefix tokens + 13,107 suffix tokens**, followed by exactly **1,024 output tokens**. Flush the cache, warm only the shared prefix, and then benchmark with `--warmup-requests 0`. Measured cache hits are **89.94%** for the listed final runs; the configured shared-prefix fraction is approximately 90%.

BF16 retains one final 32-request measurement. W8A8 retains three completed 32-request measurements. The screening rows below each use 16 requests. All listed requests completed successfully without errors.

| Selection criterion | Model / mode | C | Output token/s | Mean / P99 TPOT ms | Requests <= 50 ms |
| --- | --- | ---: | ---: | --- | --- |
| Mean and P99 <= 50 ms, final | BF16 MTP | 6 | 198.24 | 24.37 / 35.55 | 100% (32/32) |
| Mean and P99 <= 50 ms, final | W8A8 MTP | 6 | 197.75 | 24.68 / 45.48 | 100% (96/96) |
| Best mean-only screening | BF16 MTP | 6 | 200.20 | 21.21 / 29.12 | 100% (16/16) |
| Best mean-only screening | W8A8 MTP | 6 | 204.20 | 21.97 / 36.09 | 100% (16/16) |

BF16 MTP 完整启动脚本
#!/usr/bin/env bash
set -e

source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

exec python -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --tokenizer-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --context-length 139264 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-running-requests 16 \
  --max-mamba-cache-size 128 \
  --mem-fraction-static 0.80 \
  --cuda-graph-bs-decode 1 2 4 6 8 12 16 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 6688 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4


W8A8 MTP 完整启动脚本
#!/usr/bin/env bash
set -e

source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

exec python -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
  --tokenizer-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --context-length 139264 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-running-requests 16 \
  --max-mamba-cache-size 128 \
  --mem-fraction-static 0.80 \
  --cuda-graph-bs-decode 1 2 4 6 8 12 16 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 6688 \
  --quantization modelslim \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4




