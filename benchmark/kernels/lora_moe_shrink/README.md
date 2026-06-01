# Tuning the MoE LoRA Shrink Kernel

This directory auto-tunes `_moe_lora_shrink_splitk_kernel` — the LoRA A (shrink)
grouped GEMM of the merged-experts MoE LoRA path
(`python/sglang/srt/lora/triton_ops/virtual_experts.py`). The kernel projects
MoE hidden states down to the LoRA rank, one virtual expert per routed block.

## Kernel shape terminology

| Symbol | Meaning | Example values |
|--------|---------|----------------|
| `E` | num virtual experts (kernel weight dim 0) | 64, 96, 128, 192, 256, 384 |
| `N` | max LoRA rank (kernel N / output dim) | 16, 32, 64 |
| `K` | hidden size (kernel K / reduction dim) | 512, 768, 2048, 7168 |
| `M` | number of input tokens (the config key) | 1 … 8192 |

`E`, `N` and `K` select the config **file**; `M` selects the entry **inside** the file.

## What gets tuned

Each config file is split into two regimes, keyed by token count `M`:

| Regime | Pinned | Tuned |
|--------|--------|-------|
| **Decode** (small `M`) | `BLOCK_SIZE_M=16`, `BLOCK_SIZE_K=256` | `num_warps`, `num_stages`, `SPLIT_K` |
| **Prefill** (large `M`) | `BLOCK_SIZE_K=256` | `BLOCK_SIZE_M`, `num_warps`, `num_stages`, `SPLIT_K` |

`BLOCK_SIZE_N` is always the rank `N` and `BLOCK_SIZE_K` is always `256`, so
neither is tuned or stored in the config — the runtime derives both. `GROUP_SIZE_M`
is pinned to `1` (the shrink output is at most 64 wide, nothing to group along N).

Search axes: `num_warps ∈ {2,4}`, `num_stages ∈ {1,2,3,4}`, `SPLIT_K ∈
{1,2,3,4,5,6,7,8}` (clamped to `K // BLOCK_SIZE_K`), and for prefill
`BLOCK_SIZE_M ∈ {16,32,64,128}`.

Token counts `M` tuned by default (fused_moe grid):
`1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256` (decode) and
`512, 1024, 1536, 2048, 3072, 4096, 8192` (prefill).

### num_warps and num_stages bounds

- **`num_warps=1` is excluded.** With `BLOCK_SIZE_K=256`, the reduction and the
  `BLOCK_M × rank` output tile need work spread across warps; a single warp
  serializes the MMA and never wins — even at `bs=16`, where the M-tile is
  padded to 16 regardless of how many rows are real.
- **`num_stages` is bounded by shared memory.** Triton multi-buffers both
  operands, so SMEM ≈ `num_stages × (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) × 2`
  bytes (bf16/fp16). With `BLOCK_K=256`, large `BLOCK_M` exhausts the device cap
  quickly (e.g. `bm=64,bn=64,stages=4 = 256 KB > 227 KB`). The tuner drops any
  `(BLOCK_M, num_stages)` combo over the device's opt-in SMEM cap before
  benchmarking, so a tuned config always fits on the device it was tuned on. The
  runtime launcher does **not** re-check this — it only reads the config — so
  hand-edited configs must respect the cap.

## Usage

```bash
# Tune the default grid:
#   experts {64,96,128,192,256,384} x ranks {16,32,64} x hidden {512,768,2048,7168}
python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py

# Restrict the grid
python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py \
    --num-experts 128 256 --ranks 16 64 --hidden-sizes 2048 7168

# Adjust the representative MoE top_k used while benchmarking
python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py \
    --top-k 8 --dtype bfloat16
```

### Options

- `--num-experts` — expert counts `E` to tune (weight dim 0). Default: `64 96 128 192 256 384`.
- `--ranks` — LoRA ranks to tune (kernel `N`). Default: `16 32 64`.
- `--hidden-sizes` — hidden sizes to tune (kernel `K`). Default: `512 768 2048 7168`.
- `--decode-batch-sizes` — decode batch sizes tuned in the decode regime (bm pinned to 16; in decode the config key `M` == batch size, 1 token/request).
- `--prefill-num-tokens` — token counts `M` tuned in the prefill regime (bm tuned).
- `--top-k` — representative MoE top_k used while benchmarking.
- `--dtype` — `bfloat16` (default) or `float16`.

## Output

Config files are written, one per `(E, N, K, device)`, to:

```
python/sglang/srt/lora/triton_ops/moe_shrink_configs/triton_<ver>/
    moe_lora_shrink,E=<experts>,N=<rank>,K=<hidden>,device_name=<device>.json
```

Each file is a JSON object keyed by token count `M`:

```json
{
    "16":  {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 3, "SPLIT_K": 4},
    "512": {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4, "SPLIT_K": 2}
}
```

`BLOCK_SIZE_N` (= rank `N`) and `BLOCK_SIZE_K` (= 256) are derived by the runtime
and intentionally omitted from the file.

## How the configs are consumed

At runtime, `moe_lora_shrink_config.py` loads the file matching the current
`(E, N, K, device)` — where `E = weight.shape[0]` (num virtual experts), `N` the
rank, `K` the hidden size — and picks the entry for the closest tuned `M`. The
shrink launcher (`_invoke_moe_lora_shrink_splitk`) uses that config — including
the tuned `SPLIT_K` — and the same `BLOCK_SIZE_M` is used to align the MoE
routing, so the routing block size and the kernel block size stay consistent.

If no tuned file exists for the current `(E, N, K, device)`, the runtime falls
back to its heuristic default config.

The config search directory can be overridden with `SGLANG_LORA_CONFIG_DIR`
(shared with the CSGMV LoRA configs).
