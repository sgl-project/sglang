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
