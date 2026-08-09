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
rules mostly agree on what to keep — and it costs 0.5% of the denoise time. Nothing else in
this family has separated from anything else: `n_k` in {1, 2, 4, 8} at sparsity
0.75 came out within noise, and a temperature sweep of the sub-block reduction
(whose T -> infinity limit is exactly `n_k=1`) was monotone with no interior
optimum.

`SubBlockRouter.route` also accepts a `[H]` tensor as `topk` for a per-head
budget — at a fixed mean sparsity that lifts 5th-percentile recall from .52 to
.90, because diffuse heads stop being starved. The backend does not expose it
yet; there is no per-head budget to feed it.

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

MiniMax-H3 t2va, 1344x768 / 5 s / 50 steps, 8x B200, Ulysses-8, bf16. Three
prompts, one uncounted warm-up request per server, all three arms in one session
on one node. n_k=4, sparsity 0.75 (148 of 590 key blocks per query block).

| | DiT denoise | vs dense | end to end | vs dense |
| --- | ---: | ---: | ---: | ---: |
| dense (FlashAttention) | 21.48 s | 1.000x | 23.26 s | 1.000x |
| SubBlock | 18.55 s | **1.158x** | 20.00 s | 1.163x |
| SubBlock + [flashinfer#4397][fi] | 17.32 s | **1.240x** | 18.79 s | 1.238x |

Spread within an arm is under 0.07 s. End-to-end trails denoise slightly because
VAE decode is a fixed ~1.5 s that the backend does not touch.

[flashinfer-ai/flashinfer#4397][fi] rebuilds the kernel's internal Q/K/V tile
layout in one pass instead of three — `bsa_attn_blk64_fwd` was otherwise
re-tiling every activation on every call. It is bit-identical (verified
element-wise on captured H3 tensors) and **not required**: without it SubBlock
still gives 1.158x, with it 1.240x, so the patch alone is worth 1.071x.

Absolute times are node-specific — the same dense config measured 18.22 s on
another B200 host and 21.48 s here. Only ratios measured in one session are
comparable.

Sparsity is the lever, not `n_k`: at 0.80 (118 of 590 blocks) the same sweep gave
1.264x, while `n_k` moves the denoise time by 0.3% across its whole range
(17.30 s at 1, 17.35 s at 8).

[fi]: https://github.com/flashinfer-ai/flashinfer/pull/4397

`n_k` also did not resolve as a quality knob end to end. A five-prompt ablation
over `n_k` in {1, 2, 4, 8} — 1 being plain average pooling, plus a repeat of 4 as
an in-session control — reproduces the same config to 0.0006 per clip, so the
metric is precise, but the per-clip differences between arms are large and
inconsistent in sign (`n_k=1` minus `n_k=4`: -0.031, +0.057, -0.008, -0.002,
-0.013). Paired over the five clips nothing separates from `n_k=4`: t = +0.05
(n_k=1), -0.78 (n_k=2), -1.26 (n_k=8).

So the +.0142 recall the sub-block score gains over average pooling at 0.9
sparsity is real at the tensor level and does not show up in the pixels at 0.75 —
at least not above what five prompts can resolve. `n_k=4` is the default on the
recall table and on cost, not on a measured end-to-end win.

**The speedup is bounded by sequence length, not by the method.** The same
config measured 1.13x at 37.7k tokens, 1.20x at 52k and 1.47x at 96k: the
backend only touches attention, and attention's share of the DiT grows with S. Treat
1.2x as the 768p/5 s number, not the ceiling.

## Known limitation: long-range mass

At sparsity 0.75 the router keeps 82.6% of the true softmax mass, and **97% of
what it drops lies beyond 8 key blocks**. The near field survives intact; the
far field loses a sixth. On a prompt that asks for three cats this reproducibly
renders four — a duplicated adjacent object is what a long-range deficit looks
like, and the count returns only once recall reaches ~0.97 (sparsity 0.40),
which is already slower than dense. Raising the budget is not the lever; raising
*long-range* recall at a fixed budget would be.

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
