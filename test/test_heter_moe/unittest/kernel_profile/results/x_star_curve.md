# Heter-MoE speedup matrix vs. pure INT4

Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8) on A100-SXM4-80GB.
Routing: synthetic Zipf (alpha=1.1, seed=42).
Promotion policy: top-x experts by routing frequency → BF16; rest → INT4 (Marlin).
BF16 path uses **separately autotuned up + down tiles** (`bf16_sparse_configs_sep.json`), pinned via override.
INT4 path: Marlin with its built-in heuristic.
M_global rows are the 11 production-tuned batch sizes (`E=128,N=768,A100-80GB.json` keys) so both paths run on their best-tuned tile.
Each cell `xN` = `t_pure_int4 / t_at_x_N` (speedup, ≥1 means promoting N experts to BF16 helps).

| M_global | x4 | x8 | x12 | x16 | x20 | x24 | x28 | x32 | x36 | x40 | x44 | x48 | x52 | x56 | x60 | x64 | **x\*** | **T\*** | **best** | **t_int4(ms)** | **t_winner(ms)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.645 | 0.650 | 0.655 | 0.625 | 0.627 | 0.578 | 0.555 | 0.495 | 0.486 | 0.483 | 0.468 | 0.453 | 0.431 | 0.419 | 0.398 | 0.373 | **12** | **0** | **0.655×** | 0.1638 | 0.2499 |
| 64 | 0.796 | 0.840 | 0.818 | 0.785 | 0.714 | 0.689 | 0.629 | 0.616 | 0.591 | 0.569 | 0.553 | 0.522 | 0.534 | 0.519 | 0.493 | 0.453 | **8** | **16** | **0.840×** | 0.2202 | 0.2621 |
| 96 | 0.785 | 0.803 | 0.790 | 0.723 | 0.700 | 0.690 | 0.647 | 0.661 | 0.633 | 0.615 | 0.591 | 0.573 | 0.523 | 0.509 | 0.495 | 0.481 | **8** | **16** | **0.803×** | 0.2458 | 0.3062 |
| 128 | 0.684 | 0.614 | 0.614 | 0.613 | 0.606 | 0.580 | 0.575 | 0.636 | 0.619 | 0.572 | 0.554 | 0.543 | 0.529 | 0.523 | 0.510 | 0.496 | **4** | **48** | **0.684×** | 0.2775 | 0.4055 |
| 256 | 0.863 | 0.829 | 0.829 | 0.843 | 0.829 | 0.824 | 0.814 | 0.819 | 0.804 | 0.755 | 0.746 | 0.721 | 0.720 | 0.702 | 0.682 | 0.675 | **4** | **96** | **0.863×** | 0.4127 | 0.4782 |
| 512 | 0.902 | 0.909 | 0.914 | 0.940 | 0.902 | 0.895 | 0.899 | 0.895 | 0.886 | 0.868 | 0.875 | 0.871 | 0.860 | 0.823 | 0.807 | 0.785 | **16** | **48** | **0.940×** | 0.6103 | 0.6492 |
| 1024 | 1.062 | 1.081 | 1.112 | 1.201 | 1.222 | 1.233 | 1.237 | 1.235 | 1.228 | 1.224 | 1.213 | 1.228 | 1.217 | 1.214 | 1.199 | 1.184 | **28** | **48** | **1.237×** | 1.1428 | 0.9236 |
| 1536 | 0.962 | 0.969 | 1.004 | 0.990 | 0.997 | 1.029 | 1.013 | 1.003 | 1.000 | 0.991 | 0.988 | 0.990 | 0.969 | 0.971 | 0.957 | 0.951 | **24** | **80** | **1.029×** | 1.1807 | 1.1469 |
| 2048 | 0.969 | 0.989 | 1.001 | 1.000 | 1.022 | 1.021 | 1.024 | 1.051 | 1.031 | 1.032 | 1.027 | 1.038 | 1.030 | 1.045 | 1.029 | 1.023 | **32** | **80** | **1.051×** | 1.4397 | 1.3701 |
| 3072 | 1.157 | 1.205 | 1.217 | 1.230 | 1.245 | 1.252 | 1.268 | 1.270 | 1.280 | 1.307 | 1.287 | 1.293 | 1.293 | 1.292 | 1.266 | 1.266 | **40** | **96** | **1.307×** | 2.3665 | 1.8104 |
| 4096 | 1.002 | 1.004 | 1.018 | 1.034 | 1.043 | 1.050 | 1.054 | 1.059 | 1.241 | 1.263 | 1.257 | 1.262 | 1.258 | 1.262 | 1.275 | 1.263 | **60** | **80** | **1.275×** | 3.0310 | 2.3767 |

## Reading

- **x* (winner)**: optimal number of experts promoted to BF16 at this M_global.
- **T\***: per-expert-load threshold (snap-to-8) — set `bf16_promotion_threshold = T*` in the runtime config to promote the same set under this routing distribution at this M_global.
- **best speedup**: t_pure_int4 / t_at_winner_x. <1.0 means the heter-MoE path is slower than pure INT4 at this M_global (no useful promotion).
- Inner cells (`x4`..`x64`): each is the measured speedup at that specific N_active. Lets you see the shape of the curve across promotion levels per row.
