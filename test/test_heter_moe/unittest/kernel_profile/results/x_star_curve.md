# Optimal BF16/INT4 expert assignment vs. global batch size

Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8) on A100-SXM4-80GB.
Routing: synthetic Zipf (alpha=1.1, seed=42) per `scripts/heter_moe_collect_routing.py:generate_synthetic_routing`.
Promotion policy: top-x experts by routing frequency → BF16; rest → INT4 (Marlin).
BF16 tile pinned per-cell from autotuned `results/bf16_sparse_configs.json`.
Latency = paired sparse-active kernel call (fused_marlin_moe + outplace_fused_experts), median-of-50 with L2 flush + CUDA graph.

| M_global | x* (#BF16) | T* (snap) | t_layer (ms) | BF16 tile | t_pure_int4 (ms) | speedup |
|---:|---:|---:|---:|:---|---:|---:|
| 32 | 0 | 72 | 0.1454 | `n/a` | 0.1454 | 1.000× |
| 64 | 0 | 112 | 0.2212 | `n/a` | 0.2212 | 1.000× |
| 96 | 8 | 16 | 0.2847 | `n8_bse64` | 0.2929 | 1.029× |
| 128 | 0 | 224 | 0.3308 | `n/a` | 0.3308 | 1.000× |
| 192 | 8 | 40 | 0.3553 | `n8_bse128` | 0.3656 | 1.029× |
| 256 | 0 | 400 | 0.4168 | `n/a` | 0.4168 | 1.000× |
| 384 | 0 | 704 | 0.4874 | `n/a` | 0.4874 | 1.000× |
| 512 | 0 | 912 | 0.6103 | `n/a` | 0.6103 | 1.000× |
| 768 | 16 | 56 | 0.8192 | `n16_bse256` | 0.8325 | 1.016× |
| 1024 | 32 | 40 | 0.9196 | `n32_bse256` | 0.9431 | 1.026× |
| 1280 | 48 | 32 | 1.0220 | `n48_bse128` | 1.0670 | 1.044× |
| 1536 | 24 | 80 | 1.1203 | `n16_bse512` | 1.1919 | 1.064× |
| 1792 | 32 | 72 | 1.2483 | `n32_bse256` | 1.6036 | 1.285× |
| 2048 | 32 | 80 | 1.3763 | `n32_bse512` | 1.4418 | 1.048× |
| 2560 | 48 | 64 | 1.5974 | `n48_bse256` | 1.7162 | 1.074× |
| 3072 | 40 | 96 | 1.8504 | `n32_bse512` | 1.9681 | 1.064× |
| 3584 | 56 | 80 | 2.0726 | `n48_bse512` | 2.8191 | 1.360× |
| 4096 | 56 | 88 | 2.3316 | `n48_bse512` | 2.5467 | 1.092× |
| 4608 | 56 | 96 | 2.5907 | `n48_bse512` | 3.4324 | 1.325× |
| 5120 | 56 | 104 | 2.8672 | `n48_bse512` | 3.1304 | 1.092× |
| 6144 | 32 | 248 | 3.4335 | `n32_bse1024` | 3.7171 | 1.083× |
| 7168 | 24 | 384 | 4.0325 | `n16_bse2048` | 4.2916 | 1.064× |
| 8192 | 64 | 152 | 4.4308 | `n64_bse1024` | 4.8497 | 1.095× |
| 9216 | 64 | 168 | 4.9101 | `n64_bse1024` | 5.4252 | 1.105× |

## Reading the table

- **x\***: optimal number of experts to keep in BF16 at this M_global. The remaining 128−x* are kept in INT4.
- **T\* (snap)**: threshold (tokens/expert) snapped to the nearest multiple of 8 — the runtime knob `bf16_promotion_threshold` would be set to this value to promote the top-x* experts under this routing distribution at this M_global.
- **speedup**: latency reduction vs. pure-INT4 (x=0) at this M_global.
