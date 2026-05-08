# Double Sparsity benchmark

Three-way comparison driver for the v1 ship-gate (M8: Llama-3.1-8B; M9:
Llama-3.1-70B TP=8). Measures decode tok/s, TTFT, TBT (p50/p95), and a
NIAH-style retrieval probe across:

1. **`main_dense`** — clean `origin/main` checkout (regression baseline).
2. **`branch_ds_off`** — DS branch with DS disabled (proves the dense
   path is unchanged on the branch).
3. **`branch_ds_on`** — DS enabled (the speedup measurement).

The `(1)↔(2)` comparison verifies "DS off → byte-for-byte unchanged"; the
`(2)↔(3)` comparison is the actual sparse vs dense win.

## Hardware

- **M8 (8B):** 1× H200, single-node TP=1.
- **M9 (70B ship-gate):** 8× H200, TP=8 (or 4× H200 TP=4 with reduced KV).
  Numbers from any TP layout that fits the model.

## Prerequisites

A working SGLang install with the matching `sglang-kernel` version,
FA3 backend, CUDA. Generate a calibration first:

```bash
python scripts/double_sparsity/calibrate.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output calib_8b.json \
    --heavy-channels 32 \
    --n-samples 64 --seq-len 4096
```

For 70B, do the same with `--model meta-llama/Meta-Llama-3.1-70B-Instruct`.

## Running

Each config is a separate process. Easiest is a small driver shell
script that sets up the right git checkout for `main_dense`:

```bash
# 1. Clean main baseline (in a separate worktree)
git worktree add /tmp/sglang-main origin/main
cd /tmp/sglang-main
python benchmark/double_sparsity/bench_decode.py \
    --config main_dense \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --context-len 65536 --output-len 1024 \
    --output-json /tmp/results_main.json
cd -

# 2. Branch DS off
python benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_off \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --context-len 65536 --output-len 1024 \
    --output-json /tmp/results_off.json

# 3. Branch DS on
python benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_on \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --calibration calib_8b.json \
    --context-len 65536 --output-len 1024 \
    --niah --niah-context-tokens 32768 \
    --output-json /tmp/results_on.json

# 4. Compare
python benchmark/double_sparsity/compare.py \
    --main /tmp/results_main.json \
    --branch-off /tmp/results_off.json \
    --branch-on /tmp/results_on.json
```

## Ship-gate criteria (v1)

- **Speedup (DS on / DS off)**: `>= 1.30×` decode tok/s on Llama-3.1-70B
  at 64K context, output_len=1024, batch=1, on H200 (TP=8).
- **DS-off regression vs main**: `|delta| <= 2%` decode tok/s — proves
  the dense path is unchanged on the branch.
- **NIAH accuracy delta**: `>= -0.02` (DS on within 2 pts of DS off) at
  64K context.

If any check fails, ship feature-flagged off-by-default and document
the gap.

## v1.1 follow-ups (for reference)

- Replace the torch-reference selection body in `ds_select_tokens_triton`
  with the block-decomposed Triton kernel sketched in the plan.
- Add an in-graph `scheduler_metadata` recompute path if profiling
  shows the always-`None` mode is materially slower for FA3.
- Per-Q-head + per-batch union scoring (v2 path documented in the plan).

## v1 results — honest record

**Llama-3.1-8B-Instruct, single H200, bfloat16, page_size=1, FA3.**
Synthetic calibration (8 samples, 2K seq_len, S=32). Streaming
completions, batch=1, output_len=256.

| context | config | e2e tok/s | TBT p50 (ms) | DS / dense |
|---|---|---|---|---|
| 8192 | branch_ds_off | 168.7 | 5.0 | — |
| 8192 | branch_ds_on (vectorized) | 49.5 | 19.3 | 0.29× |
| 32768 | branch_ds_off | 93.8 | 5.8 | — |
| 65536 | branch_ds_off | 49.5 | 6.7 | — |
| 65536 | branch_ds_on (vectorized) | 30.4 | 19.3 | 0.61× |

**Status: v1 ship-gate not met on 8B/H200.** DS adds a constant
~13 ms of selection overhead per decode token, regardless of context
length. Dense FA3 TBT on H200 grows so slowly with context (5 ms at
8K → 6.7 ms at 64K) that DS's potential bandwidth savings can't pay
the selection cost.

**Where DS bandwidth savings can structurally show up on this
hardware/model.** Back-of-envelope; counts both K and V (FA3 reads
both during decode attention):

```
8B at 64K, bf16:
  KV cache size = 32 layers × 8 KV heads × 128 dim × 2 (K+V) × 2 B × 64K tok
                = 8 GB
  H200 HBM      = 4.8 TB/s
  KV read floor ≈ 8 GB / 4.8 TB/s ≈ 1.7 ms

  Dense TBT measured = 6.7 ms.
  → KV bandwidth is ~25% of dense TBT at this scale; the rest
    is kernel launch + Q·K compute + softmax + V GEMM. DS targets
    the bandwidth slice; the compute slice it can't help with.

  DS upper-bound saving ≈ 1.7 ms × (1 - budget/seq_len).
  At budget=1024, seq_len=64K: ≈ 1.7 × (1 - 1024/65536) ≈ 1.67 ms.
  That's the perf-path budget DS has to fit under to start winning.
```

DS-the-architecture is built to win where dense attention is more
strongly bandwidth-bound. That's longer context, larger model, or
both:

```
70B at 64K (per H200, TP=8), bf16:
  KV cache size per rank = 80 × 1 × 128 × 2 (K+V) × 2 B × 64K
                         = 2.5 GB per rank
  KV read floor          ≈ 2.5 GB / 4.8 TB/s ≈ 0.5 ms per rank
  But 80 layers compose serially; typical total dense TBT ~15–25 ms,
  with KV bandwidth a much larger share of that than at 8B/64K.
  → meaningful headroom for DS to win once selection overhead drops
    below the bandwidth slice.

70B at 128K, TP=8: KV cache per rank = 5 GB → KV-bound regime is
clearly entered. This is the real ship-gate workload.
```

**Caveat on these numbers.** The above is a rough KV-bandwidth ceiling,
not a prediction. Actual dense TBT depends on FA3's specific access
pattern (K then V), kernel-launch overhead per layer, softmax, and
non-attention work in the layer. The DS selection adds its own
bandwidth cost too — reading K_label per step. The right way to
settle this is measurement under the v1.1 fused kernel, not algebra.

**v1.1 priorities to actually ship the speedup.**
1. **Fuse the selection** into one Triton kernel that streams `K_label`
   row-by-row and accumulates per-(bs, kv_head) scores in registers.
   v1's vectorized selection has ~10 separate kernel launches per
   layer per step plus a transient `[bs, max_ctx, H_kv, S]` tensor.
   Target: cut selection overhead from ~13 ms to <1 ms.
2. **70B/H200 TP=8 bench**. This is the architecturally correct
   target where DS bandwidth savings are real even with v1's
   selection overhead.
3. **Real calibration** on a held-out long-context retrieval set for
   the accuracy probe (NIAH).

**v1 ship status.** Architecture correct, full pipeline live (server
boots, generates coherent outputs, 76 tests green including live e2e
on Qwen2.5-0.5B and Llama-3.1-8B at 8K/32K/64K). Performance gap is
purely in selection-kernel overhead; the gap is well-bounded and
v1.1's path is documented. **Ships off-by-default** until v1.1 closes
the kernel.

## v1 limitations (auto-set when --enable-double-sparsity)

- `disable_piecewise_cuda_graph = True`: v1 selection's dynamic
  gathers/scatter/sort can't be traced through torch.compile /
  breakable graph. v1.1 wraps selection as a registered split op.
- `disable_cuda_graph = False`: full CUDA graphs work. The FA3
  metadata adaptor is capture-safe, pinned by
  `test_double_sparsity_adaptor.py::TestCudaGraphCaptureReplay`.

## Memory budget reference

For Llama-3.1-70B at S=32, num_kv_heads=8 (TP=8 → 1 KV head per rank),
page_size=1, bf16 K_label:

```
K_label per token per rank = 80 layers × 1 × 32 × 2B = 5 KB
KV per token per rank      = 80 layers × 1 × 128 × 2 × 2B = 40 KB
                                               ↑ K + V
→ K_label is 12.5% overhead per rank.
For 200K tokens: K_label = 1 GB; KV = 8 GB.
```
