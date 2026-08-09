# SubBlock — training-free block-sparse attention for the MiniMax-H3 DiT

Routes FlashInfer's 64-token block-sparse kernel (`bsa_attn_blk64_fwd`) with a
sub-block score. Nothing is trained and no weights change: a cheap estimator
runs before attention and hands the kernel a `q2k_block_index`.

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend subblock \
  --attention-backend-config '{"sparsity": 0.80, "n_k": 4}'
```

Requirements come from the kernel: **SM100 (B200), bfloat16, head_dim 128**.
Anything that does not meet them — cross attention, the token refiner, short
sequences — silently runs dense, so selecting the backend is safe model-wide.

Inline JSON gets mangled by `shlex.split`; pass a **file path** to
`--attention-backend-config` if the shell eats the quotes.

## How the score works

The usual proxy for a 64x64 block is `mean(Q_block) · mean(K_block)`. Averaging
64 keys into one vector destroys exactly the variation that decides which keys a
query wants. So each block is cut into sub-blocks -- `n_k` on the key side,
`n_q` on the query side -- and every sub-block pair is scored and combined with
a log-sum-exp:

```
score(i, j) = log Σ_{a,b} exp( mean(Q_{i,a}) · mean(K_{j,b}) · softmax_scale )
```

which estimates the block's un-normalised softmax mass directly — the quantity
that says how much is lost by skipping the block. See `router.py` for the recall
table behind `n_k=4`.

Splitting the query side *alone* is worse than not splitting — a block's mass
sums over its query rows, so with one key vector to score against the query
detail averages out. Splitting both together is a different proposition, and it
is the one measurement in this family that has held up: see `n_q` below.

## Configuration

| key | default | meaning |
| --- | ---: | --- |
| `sparsity` | 0.80 | fraction of key blocks dropped per query block |
| `n_k` | 4 | key sub-blocks per 64-token block (1, 2, 4, 8) |
| `skip_first_steps` | 10 | leading denoise forwards kept dense |
| `skip_first_layers` | 0 | leading DiT blocks kept dense |
| `n_q` | 4 | query sub-blocks per 64-token block (1, 2, 4, 8) |
| `min_seq_len` | 4096 | below this the router costs more than it saves |

### `n_q`: the one estimator change that reproduced

Fifteen t2va prompts, every arm rendered in one session against that session's
own dense render, scored as cosine of the decoded video:

| | `n_q=4` over `n_q=1` | paired t | better on |
| --- | ---: | ---: | ---: |
| sparsity 0.90 | +0.062 | +2.6 | 13/15 |
| sparsity 0.75 | +0.008 | +2.1 | 10/15 |

The margin shrinks as the budget loosens — at the shipped 118 of 590 blocks the
rules mostly agree on what to keep — and it costs 0.5% of the denoise time.
Nothing else in this family has separated from anything else: see `n_k` below,
and a temperature sweep of the sub-block reduction (whose T -> infinity limit is
exactly `n_k=1`) was monotone with no interior optimum.

The one idea measured here that beats any estimator upgrade is a **per-head
budget**: at a fixed mean sparsity, spending more blocks on diffuse heads lifts
5th-percentile mass recall from .52 to .90. It is not implemented — it needs a
rule for setting the split, which nothing in the pipeline produces yet.

### Why the two schedule cutoffs are not symmetric

Swept independently on t2va 1344x768 / 5 s / 50 steps, n_k=4, sparsity 0.75,
measured as cosine against the dense render:

| | cosine vs dense | speedup |
| --- | ---: | ---: |
| skip 10 steps / 2 layers | 0.558 | 1.197x |
| skip 10 steps / 0 layers | 0.556 | 1.208x |
| skip 5 steps / 2 layers | 0.310 | 1.233x |
| skip 0 steps / 0 layers | 0.082 | 1.296x |

The early denoise steps settle the layout of the sample and re-frame the shot
when approximated; depth does not behave that way, and the layer cutoff costs
0.0013 of that cosine — inside the 0.02 run-to-run noise floor — so it defaults
to 0. Do not lower `skip_first_steps` without looking at the output.

## Measured

MiniMax-H3 t2va, 1344x768 / 5 s / 50 steps, 8x B200, Ulysses-8, bf16, at the
shipped defaults (sparsity 0.80, `n_q=n_k=4`, 118 of 590 key blocks per query
block). Five prompts per arm, all arms in one session on one node, first
(cold) sample of each arm dropped:

| | DiT denoise | vs dense |
| --- | ---: | ---: |
| dense (FlashAttention) | 18.295 s | 1.000x |
| SubBlock | 15.536 s | **1.178x** |
| SubBlock + [flashinfer#4397][fi] | 14.328 s | **1.277x** |

Spread within an arm is under 0.07 s. End to end the ratio is slightly lower —
VAE decode is a fixed ~1.5 s the backend does not touch.

[flashinfer-ai/flashinfer#4397][fi] rebuilds the kernel's internal Q/K/V tile
layout in one pass instead of three — `bsa_attn_blk64_fwd` was otherwise
re-tiling every activation on every call. It is bit-identical (verified
element-wise on captured H3 tensors) and **not required**: the patch alone is
worth 1.084x on top of the backend.

[fi]: https://github.com/flashinfer-ai/flashinfer/pull/4397

Sparsity is the speed lever and it saturates. Same methodology, `n_q=n_k=4`,
stock FlashInfer:

| | blocks kept | denoise | vs dense |
| --- | ---: | ---: | ---: |
| dense | 590/590 | 18.294 s | 1.000x |
| sparsity 0.75 | 148/590 | 16.105 s | 1.136x |
| sparsity 0.80 | 118/590 | 15.527 s | **1.178x** |
| sparsity 0.85 | 89/590 | 15.112 s | 1.211x |

Cutting the budget a further 40% past 0.80 buys 6%, and 0.85 was the worst arm
on cosine-vs-dense on both clips rendered across all three grades, so the
default takes the knee. `n_k` moves the denoise time by 0.3% across its whole
range (17.30 s at 1, 17.35 s at 8) — it is a quality knob, not a speed one.

**The speedup is bounded by sequence length, not by the method.** The same
config measured 1.13x at 37.7k tokens, 1.20x at 52k and 1.47x at 96k: the
backend only touches attention, and attention's share of the DiT grows with S.
Treat 1.2x as the 768p/5 s number, not the ceiling.

Peak memory is unchanged (99,356 vs 99,358 MiB/GPU): block sparsity saves
compute, not activations, and the `[B,H,Gq,Gk]` score matrix is ~20 MB at
S=37.7k.

Absolute times are node-specific — the same dense config measured 21.48 s on
another B200 host. Only ratios measured in one session are comparable.

### What `n_k` did *not* resolve

A five-prompt ablation over `n_k` in {1, 2, 4, 8} — 1 being plain average
pooling, plus a repeat of 4 as an in-session control — reproduces the same
config to 0.0006 per clip, so the metric is precise, but the per-clip
differences between arms are large and inconsistent in sign (`n_k=1` minus
`n_k=4`: -0.031, +0.057, -0.008, -0.002, -0.013). Paired over the five clips
nothing separates from `n_k=4`: t = +0.05 (n_k=1), -0.78 (n_k=2), -1.26
(n_k=8). The +.0142 recall the sub-block score gains over average pooling at
0.9 sparsity is real at the tensor level and does not show up in the pixels at
the shipped budget. `n_k=4` is the default on the recall table and on cost.

## Known limitation: long-range mass

At sparsity 0.75 the router keeps 82.6% of the true softmax mass, and **97% of
what it drops lies beyond 8 key blocks**. The near field survives intact; the
far field loses a sixth.

What that looks like in the pixels, on the one clip where it was chased: a
prompt asking for three cats rendered four, in both of two independent runs
where dense rendered three, and the count came back only once recall reached
~0.97 (sparsity 0.40) — a budget already slower than dense. Duplicating an
adjacent object is what a long-range deficit looks like. Treat it as a lead,
not a settled result: a later sweep of the same clip at 0.75/0.80/0.85, one
render per grade, did not reproduce the extra cat, and object counts move
between server launches at a fixed seed. Raising the budget is not the lever;
raising *long-range* recall at a fixed budget would be.

An attention sink (first 16 blocks always kept) and a forced diagonal were both
tried at fixed budget and rejected: the diagonal changed relative L2 by 0.2%,
and the sink helped only in DiT layers 2-32 and did not reach the pixels.

## Files

| | |
| --- | --- |
| `router.py` | `SubBlockRouter` — pooling, scoring, selection, `RoutingPlan` |
| `kernels.py` | Triton pooling / segmented-LSE / fused top-k |
| `../subblock_attn.py` | the `AttentionBackend`: schedule, gating, dense fallback |

Tests: `test/unit/test_subblock_attention.py`. The trick that makes the sparse
kernel checkable against dense is running it at a sparsity just above 0 — every
block is then inside the budget, so the result must reproduce dense attention up
to bf16 rounding, which pins the routing indices, the ragged tail block sizes
and the softmax scale in one assertion.
