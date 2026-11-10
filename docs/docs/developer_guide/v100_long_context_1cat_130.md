# V100 long-context lessons from 1Cat-vLLM 1.3.0

This note records which ideas from 1Cat-vLLM's `v1.2.2..v1.3.0` SM70 campaign
were independently reproduced in SGLang TileLang, which were retained, and
which remain experimental. The release commits examined were:

- v1.2.2: `644d8a7cd05ed4ecd1cd188e3c05b4bbd074f504`
- v1.3.0: `6ada86ed64af6d1a7b3cb0f34df237fd86f06d48`

Neither the host environment nor the Docker image installs 1Cat-vLLM,
FlashQLA, or zhinianqin's FlashAttention-V100 package. Attention and GDN are
SGLang-owned TileLang implementations. The only temporary 1Cat source used at
build time is a pinned sparse checkout of `LICENSE`, `csrc/sm70_turbomind`, and
`csrc/moe`, plus its required small `csrc/core` header directory, for the explicitly
retained TurboMind quantized-GEMM/MoE adapter.

## Why the ideas matter disproportionately on Volta

The useful general lesson is to optimize the dependency graph and operand
movement, not only nominal FLOPs or instruction count. Volta has no
Ampere-style `cp.async`; dependent global/shared loads are much longer than an
HMMA step, and register/shared-memory pressure easily leaves each scheduler
with too few eligible warps. Removing an instruction can still make a kernel
slower if it lengthens the critical load-to-HMMA chain.

The v1.3.0 D=256 work applied that principle in three structural ways:

1. Specialize D=256 ownership and scheduling for SM70 tensor-core fragments.
2. Gather paged KV once for a long multi-token prefill, amortizing O(K) page
   resolution across O(QK) attention work.
3. Split a long KV interval across CTAs and exactly merge FP32 online-softmax
   states, converting growing K from only serial CTA work into launch
   parallelism where the host geometry benefits.

After attention improved, projection GEMMs became the next bottleneck. That
Amdahl's-law migration is as important as any individual kernel speedup.

## Independently implemented routes

### D=256 attention and E5M2 KV

`tilelang_fa_v100` now owns the Volta attention path:

- dense D=256 prefill with one-time logical KV gathering;
- exact configurable split-KV with FP32 state merging;
- opt-in block-sparse long prefill;
- grouped GQA q=1 decode with context partitions and a separate exact reducer;
- byte-addressed E4M3/E5M2 cache support; and
- reusable FP8-to-FP16 prefill scratch buffers.

For E5M2 decode, byte-to-FP16 conversion uses exact bit expansion rather than
a dependent lookup-table load. The device chooses the active partition count
from the current sequence length and caps at 80 CTAs, matching this host's 80
SMs. Copying 1Cat's p256/p1024 thresholds or 72-SM wave geometry did not improve
this page-16 SGLang layout.

### Full SM70 GDN

The production GDN dispatcher can now select an independent TileLang backend
for packed prefill and decode. It supports variable-length packed batches,
row-strided mixed QKV input, indexed FP16/FP32 recurrent-state caches, direct
output, and column-group CTA selection. The short direct recurrence switches
to a tensor-core 64-token chunk pipeline after 1,280 tokens.

On the real FP8 checkpoint, TileLang prefill GDN reduced TTFT by 3.4% at 4K and
2.3% at 70K versus Triton. Triton remains the recommended q=1 decode backend
because GDN is a small decode bucket and its existing kernel is faster for this
shape.

### Mixed FP16/FP32 Gemma RMSNorm

The SM70 fusion performs FP16 activation plus FP32 residual update, FP32 RMS
reduction, Gemma `weight + 1`, and FP16 normalized output in one kernel. At
4,096 x 5,120 it measured 0.315 ms versus 1.397 ms for the unfused PyTorch
reference.

This result is not presented as a `Qwen/Qwen3.8-27B-FP8` model improvement.
The route only admits an actual FP32 residual, while the normal Qwen3.8 serve
path observed during the acceptance run keeps the residual in FP16.

### BFLA sparse prefill

The experimental BFLA path builds a query-head-specific coarse block selector
and skips unselected D=256 KV blocks. Q=4,096, K=32,768 at a 10% target keep
ratio measured about 9.8 ms including selection versus 23.9 ms dense.

It is disabled by default because dropping blocks changes model semantics.
All-keep mode is an exact control; approximate mode requires an additional
explicit opt-in and must pass retrieval and long-context quality evaluation.

### Greedy TP top-1

For strictly greedy, non-speculative requests, each TP rank can return only its
local top candidate instead of assembling full-vocabulary logits. The route
fails closed for sampling, speculative decoding, logprobs, penalties, grammar,
logits bias, or custom logits processors. It therefore does not affect the
primary sampling-compatible FP8 benchmark.

## FP8 acceptance result

The primary benchmark is not an FP16 proxy. It serves the real
`Qwen/Qwen3.8-27B-FP8` checkpoint on four V100-SXM2-32GB GPUs with E5M2 KV,
8,192-token chunks, one request, 256 output tokens, and speculative decoding
off.

| Input | Prefill | Decode |
| ---: | ---: | ---: |
| 1K | 2,992 tok/s | 58.2 tok/s |
| 4K | 4,137 tok/s | 57.6 tok/s |
| 25K | 3,714 tok/s | 50.8 tok/s |
| 70K | 2,980 tok/s | 38.8 tok/s |
| 128K | 2,356 tok/s | 30.0 tok/s |

The complete cold-cache protocol, TTFT/TPOT values, ablations, and the 70K
decode profile are in the
[FP8 target-only report](../../benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md).
At 70K, FP8 projection GEMMs account for about 41.7% of decode time and grouped
attention for about 40.8%. Long-context decode therefore remains a joint FP8
GEMM and attention problem; optimizing GDN alone cannot remove the decay.

## Explored ideas that were not retained as defaults

The following received working implementations and measurements:

- Scalar grouped-G6 decode was 4--5x slower than the fused tensor-core decoder.
- Staging QK and PV into separate kernels was 3.8--4.3x slower after its
  correctness issue was fixed.
- Raising the decode CTA target from 80 to 120 was slower at 128K.
- Smaller grouped-decode TileLang blocks were either slower or invalid for the
  required warp-column tensor-core layout.

Split-KV remains available for measured host-specific tuning. The important
lesson from 1Cat is to create K-axis parallelism when launch waves are
underfilled, not to hard-code three splits on every GPU. This 80-SM host does
not reproduce the exact 72-SM wave arithmetic that made split-KV3 especially
effective in the original measurements.

## Validation and source boundary

Tests cover paged and shuffled page tables, FP16 and E5M2 grouped decode,
packed GDN state/cache behavior, the exact mixed-dtype RMSNorm contract, BFLA
all-keep equivalence and approximate-mode safety gates, greedy top-1 fallback,
and speculative-decoding regressions.

The Dockerfile and host installer must preserve the source boundary stated at
the top of this document. Do not add 1Cat or third-party FlashAttention/FlashQLA
packages as installed Python dependencies; port an idea into repository-owned
TileLang code and attribute the algorithmic inspiration instead.
