# SubBlock — training-free block-sparse attention for the MiniMax-H3 DiT

Routes FlashInfer's 64-token block-sparse kernel (`bsa_attn_blk64_fwd`) with a
sub-block score. Nothing is trained and no weights change: a cheap estimator
runs before attention and hands the kernel a `q2k_block_index`.

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend subblock \
  --attention-backend-config '{"sparsity": 0.75, "n_k": 4}'
```

Inline JSON gets mangled by `shlex.split`; pass a **file path** to
`--attention-backend-config` if the shell eats the quotes.

## What it runs on

Everything below comes from `bsa_attn_blk64_fwd`, not from this backend.

| | |
| --- | --- |
| GPU | **compute capability 10.0 only** — B200 / GB200 class. The kernel is built `-gencode=arch=compute_100a,code=sm_100a`, which is arch-specific and does not forward-run on 10.3 (B300 / GB300) or 12.x (RTX PRO 6000, RTX 50xx). |
| dtype | bfloat16 |
| head_dim | 128 |
| attention | non-causal, one contiguous sequence per call |

Within a supported GPU, anything the kernel cannot serve — cross attention, the
token refiner, sequences under `min_seq_len`, non-bf16 activations, head_dim !=
128 — silently runs dense, so the backend is safe to select model-wide.

**On an unsupported GPU it is not a fallback, it is an error.** The kernel raises
`RuntimeError: BSA blk64 only supports SM100` on the first sparse attention call.
The startup resolver imports the entry point, which catches a missing or broken
FlashInfer install, but the extension itself is built lazily inside that first
call, so the device check does not happen until then.

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
| `min_seq_len` | 4096 | below this the router costs more than it saves |

**`sparsity` is an upper bound, not an exact figure.** The kernel pads each query
row's block count up to a multiple of 8 with phantom slots it then masks out, so
148 blocks costs exactly what 152 costs; the router takes the 152. At 590 blocks,
0.75 requested delivers 0.7424, and the startup log reports what was kept.

**The two schedule cutoffs are asymmetric on purpose.** The early denoise steps
settle the layout of the sample and visibly re-frame the shot when approximated —
lowering `skip_first_steps` from 10 to 5 halves cosine against the dense render.
Depth does not behave that way, so `skip_first_layers` defaults to 0. Do not
lower `skip_first_steps` without looking at the output.

## Measured

MiniMax-H3 t2va, 1344x768 / 5 s / 50 steps, 8x B200, Ulysses-8, bf16, at the
shipped defaults (152 of 590 key blocks per query block). All arms in one session
on one node, cold sample dropped; spread within an arm is under 0.07 s.

| | DiT denoise | vs dense |
| --- | ---: | ---: |
| dense (FlashAttention) | 18.270 s | 1.000x |
| SubBlock | 16.061 s | **1.138x** |
| SubBlock + [flashinfer#4397][fi] | 15.012 s | **1.217x** |

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

Peak memory is unchanged (99,356 vs 99,358 MiB/GPU): block sparsity saves
compute, not activations, and the `[B,H,Gq,Gk]` score matrix is ~20 MB at
S=37.7k. Absolute times are node-specific; only ratios measured in one session
are comparable.

## Known limitation: long-range mass

At sparsity 0.75 the router keeps 82.6% of the true softmax mass, and **97% of
what it drops lies beyond 8 key blocks**. The near field survives intact; the far
field loses a sixth. On the one clip where this was chased into the pixels, a
prompt asking for three cats rendered four where dense rendered three, and the
count came back only at a budget already slower than dense — so raising the
budget is not the lever; raising *long-range* recall at a fixed budget would be.
An attention sink and a forced diagonal were both tried at fixed budget and
rejected on measurement.

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
