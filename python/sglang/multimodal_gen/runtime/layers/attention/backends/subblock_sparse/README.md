# SubBlock sparse attention — training-free block sparsity for the MiniMax-H3 DiT

Routes the same 64-token SubBlock plan to SGLang's CuTe-DSL block-sparse
FlashAttention kernel on SM90 or FlashInfer's `bsa_attn_blk64_fwd` on SM100.
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
                               "min_seq_len": 4096}'
```

**`text_encoder=fa` is not optional.** `--attention-backend` applies to every
component, and the Qwen3-VL text encoder admits only `fa` / `torch_sdpa` /
`sage_attn_3`; without the override it raises and the server never starts. Put
the override on the *encoder*, not the DiT — `transformer=subblock_sparse_attn`
appears to work and silently does nothing, because H3 resolves the DiT backend
lazily on the first forward, outside the component-loading context that the
override applies to.

`--attention-backend-config` is optional and overrides only the keys it names,
so `'{"sparsity": 0.85}'` alone trades quality for another 6%. Inline JSON gets
mangled by `shlex.split`; pass a **file path** instead if the shell eats the
quotes.

## What it runs on

The backend selects an architecture-specific kernel; their shared constraints
are listed below.

| | |
| --- | --- |
| GPU | **compute capability 9.0 or 10.0** — H100 / H200 use SGLang's CuTe-DSL SM90 block-sparse FlashAttention kernel; B200 / GB200 use FlashInfer's architecture-specific `sm_100a` kernel. Other capabilities, including 10.3 (B300 / GB300) and 12.x (RTX PRO 6000, RTX 50xx), are rejected. |
| dtype | bfloat16 |
| head_dim | 128 |
| attention | non-causal, one contiguous sequence per call |

Inside the DiT, anything the kernel cannot serve — cross attention, the token
refiner, sequences under `min_seq_len`, non-bf16 activations, head_dim != 128 —
falls back to dense for that call, so no layer has to be excluded by hand.

**On an unsupported GPU it is not a fallback, it is an error at startup.** The
resolver accepts exactly compute capability 9.0 or 10.0 before loading either
kernel, so a B300 or an SM12x GPU fails at launch rather than after ten dense
denoise steps. The exact 10.0 check is required because FlashInfer's kernel is
built for `sm_100a` and has no forward-compatible 10.3 cubin.

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
| `n_k` | 4 | key sub-blocks per 64-token block (1, 2, 4, 8) |
| `n_q` | 4 | query sub-blocks per 64-token block (1, 2, 4, 8) |
| `skip_first_steps` | 10 | leading denoise forwards kept dense |
| `skip_first_layers` | 0 | leading DiT blocks kept dense |
| `min_seq_len` | 4096 | shorter sequences run dense |

**`sparsity` is an upper bound, not an exact figure.** The kernel pads each query
row's block count up to a multiple of 8 with phantom slots it then masks out, so
148 blocks costs exactly what 152 costs; the router takes the 152. At 590 blocks,
0.75 requested delivers 0.7424, and the startup log reports what was kept. It is
the speed lever — see below — and the only knob most users should touch.

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
| `../subblock_sparse_attn.py` | the `AttentionBackend`: schedule, gating, dense fallback |

Tests: `test/unit/test_subblock_sparse_attention.py`. The trick that makes the sparse
kernel checkable against dense is running it at a sparsity just above 0 — every
block is then inside the budget, so the result must reproduce dense attention up
to bf16 rounding, which pins the routing indices, the ragged tail block sizes
and the softmax scale in one assertion.
