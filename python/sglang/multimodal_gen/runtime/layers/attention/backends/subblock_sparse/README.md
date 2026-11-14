# SubBlock sparse attention — training-free block sparsity for the MiniMax-H3 DiT

Routes a SubBlock plan to SGLang's CuTe-DSL block-sparse FlashAttention kernel
on SM90 or FlashInfer's architecture-specific blk64 kernels on SM100 and SM120.
The architecture-neutral `"compute_mode": "sage_fp8"` selects online Sage-style
FP8 compute; its current SM90 implementation uses the native SageAttention2
INT8-QK/FP8-PV Hopper kernel. SM120 uses FlashInfer's CAKE Sage kernel
with Q64 x K64 blocks. SM100 supports BF16 only.
Nothing is trained and no weights change: a cheap estimator runs before
attention and hands the selected kernel a `q2k_block_index`.

Spelled out in full, with every key at its default — which is the recommended
configuration and what every number below was measured at:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 --performance-mode speed \
  --attention-backend subblock_sparse_attn \
  --component-attention-backends text_encoder=fa \
  --attention-backend-config '{"sparsity": 0.75, "n_k": 4, "n_q": 4,
                               "skip_first_steps": 10, "skip_first_layers": 0,
                               "min_seq_len": 4096, "compute_mode": "bf16"}'
```

**The text-encoder override is not optional.** `--attention-backend` applies to
every component, and the Qwen3-VL text encoder admits only `fa`, `torch_sdpa`,
or `sage_attn_3`; without the override it raises and the server never starts.
Put the override on the *encoder*, not the DiT —
`transformer=subblock_sparse_attn` appears to work and silently does nothing,
because H3 resolves the DiT backend lazily on the first forward, outside the
component-loading context that the override applies to.

On SM120, use `text_encoder=torch_sdpa` instead. The CUDA platform selects
Torch SDPA for dense attention on SM12.x, and component-specific backend
requests are validated strictly.

`--attention-backend-config` is optional and overrides only the keys it names,
so `'{"sparsity": 0.85}'` alone trades quality for another 6%. Inline JSON gets
mangled by `shlex.split`; pass a **file path** instead if the shell eats the
quotes.

## What it runs on

The backend selects an architecture-specific kernel; their shared constraints
are listed below.

| | |
| --- | --- |
| GPU | **compute capability 9.0, 10.0, or 12.0** — H100 / H200 use SGLang's CuTe-DSL SM90 block-sparse FlashAttention kernel or the current `sage_fp8` implementation; B200 / GB200 use FlashInfer's architecture-specific `sm_100a` BF16 kernel; SM120 devices use FlashInfer's CuTe-DSL BF16 kernel or CAKE Sage FP8 kernel. Other capabilities, including 10.3 (B300 / GB300), are rejected. |
| dtype | bfloat16 |
| head_dim | 128 |
| attention | non-causal, one contiguous sequence per call |

Inside the DiT, anything the kernel cannot serve — cross attention, the token
refiner, sequences under `min_seq_len`, non-bf16 activations, head_dim != 128 —
falls back to dense for that call, so no layer has to be excluded by hand.

**On an unsupported GPU it is not a fallback, it is an error at startup.** The
resolver accepts exactly compute capability 9.0, 10.0, or 12.0 before loading
the selected kernel, so a B300 or another unsupported capability fails at
launch rather than after ten dense denoise steps. The exact 10.0 check is
required because FlashInfer's kernel is built for `sm_100a` and has no
forward-compatible 10.3 cubin.

## How the score works

The usual proxy for a 64x64 block is `mean(Q_block) · mean(K_block)`. Averaging
64 keys into one vector destroys exactly the variation that decides which keys a
query wants. So each block is cut into sub-blocks — `n_k` on the key side, `n_q`
on the query side — and every sub-block pair is scored and combined with a
log-sum-exp:

```
score(i, j) = log Σ_{a,b} exp( mean(Q_{i,a}) · mean(K_{j,b}) · softmax_scale )
```

which estimates the block's un-normalised softmax mass directly — the quantity
that says how much is lost by skipping the block.

Splitting the query side *alone* is worse than not splitting: a block's mass sums
over its query rows, so with one key vector to score against the query detail
averages out. Splitting both together is a different proposition, and the only
estimator change in this family that has held up end to end. `router.py` carries
the recall table behind `n_q = n_k = 4` and the record of what was tried and
rejected.

## Configuration

| key | default | meaning |
| --- | ---: | --- |
| `sparsity` | 0.75 | key blocks dropped per query block, as an upper bound |
| `n_k` | 4 (`sage_fp8` on SM90: 8) | key sub-blocks per key block (1, 2, 4, 8) |
| `n_q` | 4 | query sub-blocks per 64-token block (1, 2, 4, 8) |
| `skip_first_steps` | 10 | leading denoise forwards kept dense |
| `skip_first_layers` | 0 | leading DiT blocks kept dense |
| `min_seq_len` | 4096 | shorter sequences run dense |
| `compute_mode` | `"bf16"` | compute path: `"bf16"` or architecture-dispatched `"sage_fp8"` |
| `quantize_before_a2a` | `false` | experimental H3 SM120 prepared Sage2 exchange over NCCL; requires `sage_fp8` and newer FlashInfer |

`compute_mode="sage_fp8"` is the stable public name, not a promise that every
architecture quantizes every operand identically. On SM90 it calls
SpargeAttention's native Hopper SageAttention2 kernel: Q/K are quantized online
to INT8 and V/P to E4M3. Its Q64 x K128 geometry defaults `n_k` to 8, preserving
the normal router's 16-token key pooling cells. Install the optional dependency:

```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

SM120 uses FlashInfer's CAKE backend with caller-owned contiguous BF16 output
and the online quantizer
`quantize_sage_qkv_sm120`, added in
[FlashInfer #4691](https://github.com/flashinfer-ai/flashinfer/pull/4691),
which the default FlashInfer install includes.
The quantizer reads strided BHSD views, avoiding BF16 Q/K/V copies before
quantization. The adapter returns a BSHD view of CAKE's output; the packed
document path writes that view directly into its destination buffer.
Sparse block indices and per-row counts are made contiguous for CAKE.

Enable it on SM90 or SM120 with:

```bash
--attention-backend-config '{"compute_mode":"sage_fp8"}'
```

**`sparsity` is an upper bound, not always an exact figure.** BF16 plans round
the budget up to groups of 8: 148 blocks costs what 152 costs. Native SM90
`sage_fp8` uses 128-token key blocks without that rounding; at S=37,760 and
sparsity 0.75 it keeps 74 of 295 blocks. Startup logs report the actual budget.

**`n_k` and `n_q` buy score accuracy, not speed.** They set how finely a block is
cut before scoring: `n_k=4` means four 16-token key sub-blocks, and the block's
score is the log-sum-exp over all `n_q * n_k` sub-block pairs. Raising them
sharpens the estimate of which blocks carry mass, at `n_q * n_k` times the score
matrix — 0.5% of denoise time at the default, so cost is not the constraint.
Raise `n_q` and `n_k` **together**: splitting the query side alone is worse than
not splitting at all.

**The two schedule cutoffs are asymmetric on purpose.** `skip_first_steps` keeps
the leading denoise forwards dense; those steps settle the layout of the sample
and visibly re-frame the shot when approximated — lowering it from 10 to 5 halves
cosine against the dense render. Depth does not behave that way, so
`skip_first_layers` defaults to 0 and every DiT layer is sparse. Do not lower
`skip_first_steps` without looking at the output.

**`min_seq_len` is a floor, not a tuning knob.** Below it the whole call runs
dense, and in packed varlen batches the test is per document, so H3's padding
tail goes dense while the 37.7k-token media document is routed. Two things break
down on short sequences: the router is four fixed Triton launches against an
attention cost that falls as S², so the overhead stops paying for itself; and the
budget goes coarse — 4096 keys is only 64 blocks, and at 1024 keys the
multiple-of-8 floor already keeps half of them. 4096 sits well below any real
video sequence and well above where either effect bites; it was chosen on that
reasoning rather than from a measured threshold sweep.

## Measured

MiniMax-H3 t2va, 1344x768 / 5 s / 50 steps, 8x B200, Ulysses-8, bf16, at the
shipped defaults (152 of 590 key blocks per query block). All arms in one session
on one node, cold sample dropped; spread within an arm is under 0.07 s.

| | DiT denoise | vs dense |
| --- | ---: | ---: |
| dense (FlashAttention) | 18.270 s | 1.000x |
| SubBlock sparse | 16.061 s | **1.138x** |
| SubBlock sparse + [flashinfer#4397][fi] | 15.012 s | **1.217x** |

[flashinfer-ai/flashinfer#4397][fi] rebuilds the kernel's internal Q/K/V tile
layout in one pass instead of three. It is bit-identical and **not required**:
worth 1.070x on its own.

[fi]: https://github.com/flashinfer-ai/flashinfer/pull/4397

Sparsity is the speed lever and it saturates — 0.75 gives 1.136x, 0.80 gives
1.178x, 0.85 gives 1.211x. Cutting the budget 40% past 0.75 buys 6%, because at
37.7k tokens attention is no longer the bulk of the step, and 0.85 rendered worst
of the three on cosine against dense. `n_k` moves the denoise time by 0.3% across
its whole range: it is a quality knob, not a speed one.

**The speedup is bounded by sequence length, not by the method.** The same config
measured 1.13x at 37.7k tokens, 1.20x at 52k and 1.47x at 96k — the backend only
touches attention, and attention's share of the DiT grows with S. Treat 1.2x as
the 768p/5 s number, not the ceiling.

The same effect shows up in the sequence-parallel degree, since that sets how
much of the sequence each GPU holds: on 4x B200 at Ulysses-4 the identical
config gives **1.168x** on denoise and **1.138x** end to end, against 1.138x on
denoise at Ulysses-8.

Peak memory is unchanged (99,356 vs 99,358 MiB/GPU): block sparsity saves
compute, not activations, and the `[B,H,Gq,Gk]` score matrix is ~20 MB at
S=37.7k. Absolute times are node-specific; only ratios measured in one session
are comparable.

## Files

| | |
| --- | --- |
| `router.py` | `SubBlockRouter` — pooling, scoring, selection, `RoutingPlan` |
| `kernels.py` | Triton pooling / segmented-LSE / fused top-k |
| `../../../../../../kernels/ops/attention/subblock_sage_fp8_sm90.py` | production adapter to native SM90 SageAttention2 |
| `../subblock_sparse_attn.py` | the `AttentionBackend`: schedule, gating, dense fallback |

Tests: `test/unit/test_subblock_sparse_attention.py` and
`test/registered/kernels/ops/attention/test_subblock_sage_fp8_sm90.py`. The GPU
test covers the native production dispatch. Running at a full block budget must
reproduce dense attention up to the expected quantization error, pinning routing
indices, ragged tails, scale domains and the softmax scale in one check.

Run the SM120 Sage numerical test with:

```bash
python test/manual/attention/test_subblock_sage_fp8_sm120.py
```

## Experimental H3 quantization before Ulysses (SM120)

`quantize_before_a2a` defaults to `false`. An explicit opt-in uses FlashInfer's
prepared Sage2 API to quantize local Q/K/V before NCCL all-to-all, then feeds
INT8 Q/K, FP8 V and their scales directly to the SM120 CAKE sparse kernel:

```bash
--attention-backend subblock_sparse_attn \
--component-attention-backends text_encoder=torch_sdpa \
--attention-backend-config '{"compute_mode":"sage_fp8","quantize_before_a2a":true}'
```

This draft path needs FlashInfer containing
[#4924](https://github.com/flashinfer-ai/flashinfer/pull/4924),
[#4951](https://github.com/flashinfer-ai/flashinfer/pull/4951), and
[#5083](https://github.com/flashinfer-ai/flashinfer/pull/5083).
The required interfaces are available in
[FlashInfer 0.7.4](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.4).
SGLang's pinned FlashInfer **0.7.3 does not include `prepare_qkv`**. The option
raises an actionable error with that release; no dependency pin is changed
here. Validating and aligning the dependency and JIT-cache pins is required
before this becomes ready to merge. Prepared exchange uses **NCCL**; the public FlashInfer API rejects
prepared PCIe/RDMA communicators, so ordinary PCIe transport support does not
make prepared low-precision RDMA available.

`backend="nccl"` chooses the communicator backend. The low-precision pipeline
is selected separately by `prepare_qkv(..., quantization="sage2")` and executed
by `scatter_qkv()`: local statistics, FP32 statistics AllGather, quantize/pack,
then the low-precision QKV AllToAll. Ordinary `scatter_heads()` is unchanged.

Communication and attention are separate components:

- `runtime/distributed/sage_ulysses.py` owns prepared exchange, cached transport
  buffers and output gather. Its `prepared_sage_ulysses_forward` accepts an
  `attention` callable and passes FlashInfer's `UlyssesQKV` record directly to
  it, including the operand layouts, scales and live/logical sequence lengths.
- `prepared_sage_attention.py` implements `PreparedSubBlockSageAttention`,
  which consumes that record, performs SubBlock routing and CAKE compute, and
  returns floating output in the global BSHD/head-sharded layout for gather.
  Its H3 wrapper owns schedule, geometry and query-mask admission.

The communication wrapper has no dependency on SubBlock or CAKE.
`backends/prepared_sage_attn.py` also provides `PreparedDenseSageAttention`.
It calls SageAttention2's native INT8-QK/FP8-PV per-warp kernel directly with
NHD views of the live Q/K prefix, the received permuted V and all three scale
tensors. It uses FP16 short-term accumulation with an FP32 buffer, matching
FlashInfer's Sage2 V range of 2.25. There is no BF16 reconstruction,
requantization, SubBlock routing or CAKE layout packing in the dense consumer.
Its output matches the record's global shape and `input_dtype`; padding rows
are cleared before gather.
The native Sage extension launches on CUDA's default stream. The dense
consumer bridges a different incoming compute stream before and after that
launch, preserving the ordering between prepared exchange, attention and
output gather without synchronizing the device.

For dense Sage on the H3 transformer, enable:

```bash
--attention-backend sage_attn \
--component-attention-backends text_encoder=torch_sdpa \
--attention-backend-config '{"quantize_before_a2a":true}'
```

The dense option defaults to `false`. It requires SageAttention built for
SM120 with
`sageattention.sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`.
The optional kernel is resolved before prepared QKV exchange. The TRT-LLM
Gen Sage implementation cannot be combined with this opt-in because it uses
a different quantization contract. Causal/dropout calls and unsupported
geometry retain the existing path before prepared exchange. Dense Sage has
no SubBlock routing-metadata limit. Both consumers share transport workspaces,
geometry replacement and teardown. The dense CUDA/multi-GPU validation entry
point below has not been run for this revision.

Supported calls are inference on SM120, head dimension 128, Ulysses degree 4
or 8, and one live packed H3 document followed by padding. Ring, training,
graph capture, unsupported geometry and oversized routing metadata decline
before prepared QKV exchange. On SM120, `compute_mode="sage_fp8"` uses dense
SageAttention2 for DiT warmup steps, skipped DiT layers and short dense DiT
calls. With `quantize_before_a2a=true`, these calls use the dense prepared
consumer and subsequent sparse steps use SubBlock/CAKE over the same
transport. Without prepared exchange they use the dense Sage backend after
the ordinary BF16 QKV exchange. Non-DiT token-refiner calls keep their dense
backend. Once collective preparation starts, failures propagate. The option
does not change the existing `quality=high` admission rules.

Before exchange, the transport clears disposable padded Q/K/V projection rows
in place as required by the prepared API. Live rows and projection-view strides
are preserved; this keeps global K statistics independent of padding.

Serialized inference retains only the current prepared geometry for each
process group/device. Matching shapes, dtypes and live lengths reuse the same
workspace and receive/output buffers across layers, steps and requests. A
change releases the old buffers before preparing a replacement, so serving
many resolutions or prompt lengths does not accumulate workspaces or hit a
lifetime entry limit. Failed initialization is not cached, and group teardown
releases the current transport. Returned output tensors keep their storage
valid after a replacement through PyTorch ownership.

Routing uses a **quantized-operand estimator**, including K tile scales but
omitting Q scales and K channel means. It can select different blocks from the
native BF16 router. This path needs its own H3 quality and latency evaluation;
no output equivalence or end-to-end speedup is claimed for the current main.

Triton pre/post fusion packs the CAKE layouts and scales, pools the quantized
operands without full BF16 Q/K materialization, expands mixed dense/sparse
query rows, and writes the output while zeroing padding. Set
`SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST=0` to compare with the unfused reference
using the **same quantized-operand estimator**. Fusion equivalence does not
establish equivalence to BF16 routing.

Validation entry points:

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_prepared_sage.py python/sglang/multimodal_gen/test/unit/test_cake_prepost_kernels.py python/sglang/multimodal_gen/test/unit/test_prepared_dense_sage.py -q
torchrun --standalone --nproc_per_node=4 test/manual/attention/test_h3_prepared_sage_ulysses.py
torchrun --standalone --nproc_per_node=4 test/manual/attention/test_h3_prepared_dense_sage_ulysses.py
H3_BENCH_RESULT=result.json torchrun --standalone --nproc_per_node=4 test/manual/attention/bench_h3_prepared_sage_ulysses.py
```

The distributed check exercises strided QKV views, nonaligned shards, live
tails, mixed query masks, persistent receive buffers and the output gather. It
compares fused and unfused glue exactly; full H3 generation still needs a
separate validation run.

The benchmark uses H3's 56 heads, D=128 and a 37,765-token live sequence with
strided QKV views and mixed dense/sparse query rows. It checks exact output
agreement, warms both arms, then alternates fused/unfused measurements over
six trials of ten calls. Reported latency is the median of the maximum-rank
CUDA event time per trial. It measures prepared attention with the same
quantized-operand estimator; it does not compare BF16 routing or establish an
end-to-end model speedup. Run on otherwise idle GPUs.
