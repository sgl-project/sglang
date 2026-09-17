# Qwen3.8 QSA packed-varlen decode for SM121

This implementation was optimized by Codex and Kimi K3 agents through
[KDA-1.5](https://github.com/radixark/KDA-1.5). The task and immutable real
tensor replay were registered in [radixark/KDA-1.5 PR #4](https://github.com/radixark/KDA-1.5/pull/4)
at commit `414ce456e14ae8546f77d9356d2c4d955c5bb7f1`. This package integrates
winning submission `b4181149c8884ddb`; its byte-exact submitted source has SHA256
`4f9977f88abfea4393a2add3a2c9255699f7e13b981dbc1a976b024b3b00e909`.

The kernel is specialized for the packed QSA decode tensors captured from
`RadixArk/Qwen3.8-Flash-Next-NVFP4` on NVIDIA GB10 (SM121):

- BF16 query, key, value, and output with head dimension 256
- one packed query row per sequence and device-side `cu_seqlens`
- 12 query heads per KV head: TP1 uses 24Q/2KV and TP2 uses 12Q/1KV
- all query-row counts in the validated `1 <= bs <= 128` envelope
- `max_seqlen_k` capacity up to 2055 and captured logical selected-KV lengths
  up to 2051 rows per sequence

The implementation groups the 12 query heads that share one KV head into one
CTA, uses BF16 tensor-core QK/PV products with FP32 online-softmax state, and
splits long KV rows across multiple CTAs. The last arriving split performs a
stable FP32 merge and resets its device counter in the same launch. A
host-visible shape/topology policy selects the two measured schedules, while
the live device `cu_seqlens_k` selects one, two, four, or eight active splits
without a host synchronization.

SM121 dispatch checks the exact Qwen3.8 contract and routes directly to this
kernel; it is the only packed-QSA attention implementation added by this PR.
The KDA replay passed all 15 TP1/TP2 production tensors on two independent GB10
GPUs, and the final source passed 150,000 consecutive launches with all
counters returning to zero.

After adaptation into SGLang, the packaged kernel passed the same 15/15 replay
with exactly one CUDA activity per row and a 2.0702x all-shape geomean over the
generic Triton fallback (1.6951x large, 2.3653x small). On one DGX Spark running
the full TP1 NVFP4 model with NEXTN, three-round low-concurrency serving A/B
improved total token throughput by 4.45% at concurrency 1 and 4.00% at
concurrency 4. A 50-example, five-shot GSM8K A/B with a 2048-token output limit
scored 49/50 for both Triton and KDA, with the same single failed example.

An additional synthetic GB10 sweep covers both TP topologies, every batch size
from 1 through 16, and short plus saturated KV rows. All 64 cases passed; the
maximum relative L2 against the original correct Triton implementation was
0.002422, and speedup ranged from 1.41x to 5.09x.

A follow-up extended-batch sweep covers both TP topologies, batch sizes
17/24/32/48/64/96/128, and short, medium, plus saturated KV rows. All 42 cases
passed with maximum relative L2 0.002410. Geomean speedup was 4.48x, the slowest
case still improved by 1.58x, and no case regressed. The packaged scratch space
is therefore sized for the largest tested batch, 128.
