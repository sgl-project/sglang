# E2E ShareGPT — heter-MoE (treatment) vs pure INT4 (baseline)

Qwen3-30B-A3B-GPTQ-Int4 weights + heter_config (top-79/layer dual-loaded BF16+INT4 by activation_frequency rank).
EfficiencyPromotionPolicy = precomputed M→expert-mask lookup; BF16 tiles pinned per-cell from `bf16_sparse_configs_sep.json` (K=5 refined).
Both servers TP=1 on A100-80GB, **PCG enabled on both**. ShareGPT, 300 prompts, mc ∈ {32, 64, 128}.

Δ = (treatment - baseline) / baseline. Throughput: positive = treatment better. Latency: negative = treatment better.

| mc | metric | baseline | treatment | Δ |
|---:|:---|---:|---:|---:|
| 32 | TPS (req/s) | 7.016 | 7.043 | +0.4% |
| 32 | output throughput (tok/s) | 1421.6 | 1427.0 | +0.4% |
| 32 | median TTFT (ms) | 62.0 | 66.3 | +6.9% |
| 32 | mean TTFT (ms) | 106.4 | 115.9 | +8.9% |
| 32 | p99 TTFT (ms) | 533.1 | 611.7 | +14.7% |
| 32 | median ITL (ms) | 15.69 | 15.03 | -4.2% |
| 32 | mean ITL (ms) | 19.59 | 19.63 | +0.2% |
| 32 | p99 ITL (ms) | 83.74 | 91.96 | +9.8% |
| 64 | TPS (req/s) | 7.864 | 8.110 | +3.1% |
| 64 | output throughput (tok/s) | 1593.4 | 1643.2 | +3.1% |
| 64 | median TTFT (ms) | 76.4 | 82.9 | +8.5% |
| 64 | mean TTFT (ms) | 293.4 | 231.3 | -21.2% |
| 64 | p99 TTFT (ms) | 1396.1 | 1063.7 | -23.8% |
| 64 | median ITL (ms) | 22.30 | 21.34 | -4.3% |
| 64 | mean ITL (ms) | 27.91 | 27.81 | -0.3% |
| 64 | p99 ITL (ms) | 111.85 | 120.62 | +7.8% |
| 128 | TPS (req/s) | 8.753 | 9.000 | +2.8% |
| 128 | output throughput (tok/s) | 1773.4 | 1823.7 | +2.8% |
| 128 | median TTFT (ms) | 131.6 | 141.1 | +7.2% |
| 128 | mean TTFT (ms) | 1288.6 | 1485.2 | +15.3% |
| 128 | p99 TTFT (ms) | 3941.5 | 4252.5 | +7.9% |
| 128 | median ITL (ms) | 25.17 | 24.07 | -4.3% |
| 128 | mean ITL (ms) | 34.22 | 33.50 | -2.1% |
| 128 | p99 ITL (ms) | 156.66 | 168.84 | +7.8% |

## Headlines (mc=64, mid-range)

- **TPS +3.1%** (7.86 → 8.11 req/s)
- **mean TTFT -21.2%** (293 → 231 ms)
- **p99 TTFT -23.8%** (1396 → 1064 ms)
