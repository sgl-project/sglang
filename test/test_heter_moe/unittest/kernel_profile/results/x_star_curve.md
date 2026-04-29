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
| 32 | 0.723 | 0.729 | 0.713 | 0.685 | 0.651 | 0.607 | 0.587 | 0.513 | 0.505 | 0.510 | 0.482 | 0.468 | 0.440 | 0.431 | 0.406 | 0.376 | **8** | **8** | **0.729×** | 0.1628 | 0.2232 |
| 64 | 0.875 | 0.895 | 0.875 | 0.809 | 0.760 | 0.737 | 0.663 | 0.647 | 0.631 | 0.612 | 0.595 | 0.579 | 0.565 | 0.551 | 0.530 | 0.495 | **8** | **16** | **0.895×** | 0.2437 | 0.2724 |
| 96 | 0.780 | 0.803 | 0.778 | 0.720 | 0.699 | 0.692 | 0.653 | 0.636 | 0.609 | 0.588 | 0.572 | 0.557 | 0.516 | 0.500 | 0.487 | 0.477 | **8** | **16** | **0.803×** | 0.2468 | 0.3072 |
| 128 | 0.800 | 0.722 | 0.720 | 0.722 | 0.717 | 0.773 | 0.762 | 0.744 | 0.722 | 0.670 | 0.650 | 0.636 | 0.620 | 0.614 | 0.597 | 0.581 | **4** | **48** | **0.800×** | 0.3246 | 0.4055 |
| 256 | 0.862 | 0.837 | 0.837 | 1.029 | 1.008 | 1.008 | 0.994 | 1.000 | 0.984 | 0.925 | 0.908 | 0.882 | 0.879 | 0.859 | 0.834 | 0.827 | **16** | **24** | **1.029×** | 0.5038 | 0.4895 |
| 512 | 0.906 | 0.910 | 1.008 | 1.146 | 1.103 | 1.096 | 1.101 | 1.096 | 1.085 | 1.061 | 1.074 | 1.067 | 1.052 | 1.008 | 0.985 | 0.963 | **16** | **48** | **1.146×** | 0.7475 | 0.6523 |
| 1024 | 1.017 | 1.081 | 1.218 | 1.205 | 1.229 | 1.239 | 1.241 | 1.241 | 1.233 | 1.230 | 1.218 | 1.234 | 1.222 | 1.220 | 1.203 | 1.188 | **28** | **48** | **1.241×** | 1.1448 | 0.9226 |
| 1536 | 1.147 | 1.181 | 1.220 | 1.205 | 1.214 | 1.252 | 1.233 | 1.218 | 1.214 | 1.204 | 1.200 | 1.205 | 1.190 | 1.187 | 1.172 | 1.157 | **24** | **80** | **1.252×** | 1.4356 | 1.1469 |
| 2048 | 1.182 | 1.208 | 1.219 | 1.230 | 1.244 | 1.256 | 1.264 | 1.293 | 1.278 | 1.275 | 1.284 | 1.290 | 1.261 | 1.267 | 1.256 | 1.247 | **32** | **80** | **1.293×** | 1.7469 | 1.3507 |
| 3072 | 0.967 | 1.003 | 1.016 | 1.025 | 1.035 | 1.034 | 1.053 | 1.060 | 1.055 | 1.072 | 1.058 | 1.062 | 1.063 | 1.068 | 1.054 | 1.065 | **40** | **96** | **1.072×** | 1.9743 | 1.8422 |
| 4096 | 0.954 | 0.978 | 0.989 | 1.009 | 1.019 | 1.026 | 1.028 | 1.026 | 1.034 | 1.043 | 1.039 | 1.042 | 0.884 | 1.051 | 0.894 | 1.044 | **56** | **88** | **1.051×** | 2.5057 | 2.3839 |

## Reading

- **x* (winner)**: optimal number of experts promoted to BF16 at this M_global.
- **T\***: per-expert-load threshold (snap-to-8) — set `bf16_promotion_threshold = T*` in the runtime config to promote the same set under this routing distribution at this M_global.
- **best speedup**: t_pure_int4 / t_at_winner_x. <1.0 means the heter-MoE path is slower than pure INT4 at this M_global (no useful promotion).
- Inner cells (`x4`..`x64`): each is the measured speedup at that specific N_active. Lets you see the shape of the curve across promotion levels per row.
