# GLM-5.3-Flash KDA qkv PTPC validation

This report validates the opt-in qkv-only PTPC path on the complete
GLM-5.3-Flash Day-0 enablement stack.

- Enablement base: `fb57dfc032`
- Tested PTPC head: `e7f884f500`
- AITER: `d9e5ef7ce0`
- Hardware: MI355X (`gfx950`)
- Source checkpoint: `zai-org/GLM-5.3-Flash@03eb536628`
- Quark checkpoint: `amd/GLM-5.3-Flash-Quark-MXFP4@b5688f2549`
- Enable: `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj`
- Roll back: unset `SGLANG_OPT_GLM53_KDA_PTPC_MODULES`

## Selected scope

Calibration rejected KDA `o_proj`: it regressed every TP8 shape, including
3.7% at M=131072. Only packed KDA `qkv_proj` remains eligible.

The calibrated crossover is `M > 4095`. M=4095 and below retain AITER BF16;
M=4096 and above use per-token activation/per-channel weight FP8.

The original BF16 parameter remains registered. The bpreshuffled FP8 weight
and scale are non-persistent buffers, so device/lifecycle operations see them
but checkpoint state does not.

## Component correctness and timing

The registered CPU/MI355X tests cover default-off behavior, all enablement
predicates, invalid weights, repack/shuffle ownership, non-persistent buffer
lifecycle, BF16 rollback, fused-topology rejection, unknown selectors, zero
tokens, TP4/TP8 shapes, the M=4095/4096 boundary, deterministic replay, finite
output, cosine similarity, and mean error.

Result: **12 tests passed, 37 subtests passed**.

The committed benchmark includes activation quantization in PTPC time. Source
checkpoint results use ten operations per timed sample:

| TP | M | BF16 ms | PTPC ms | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4096 | 0.1588 | 0.1111 | -30.05% |
| 4 | 8192 | 0.2748 | 0.2061 | -25.01% |
| 4 | 16384 | 0.5371 | 0.4089 | -23.87% |
| 4 | 131072 | 4.2298 | 3.3190 | -21.53% |
| 8 | 4096 | 0.0815 | 0.0629 | -22.79% |
| 8 | 8192 | 0.1587 | 0.1139 | -28.21% |
| 8 | 16384 | 0.2722 | 0.2183 | -19.78% |
| 8 | 131072 | 2.1420 | 1.8524 | -13.52% |

The Quark checkpoint independently reproduces the result:

| TP | M | BF16 ms | PTPC ms | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8192 | 0.2966 | 0.2281 | -23.09% |
| 4 | 16384 | 0.5609 | 0.4333 | -22.74% |
| 4 | 131072 | 4.2385 | 3.3384 | -21.24% |
| 8 | 8192 | 0.1803 | 0.1321 | -26.74% |
| 8 | 16384 | 0.2899 | 0.2401 | -17.18% |
| 8 | 131072 | 2.1604 | 1.8814 | -12.92% |

## Accuracy

Full 1,319-example GSM8K, thinking enabled, temperature 1.0, top-p 0.95,
seed 0, decode CUDA graphs active:

| TP | BF16 | qkv PTPC | Delta | PTPC errors |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 97.04% | 96.89% | -0.15 pp | 0.00% |
| 8 | 97.12% | 96.89% | -0.23 pp | 0.00% |

All four arms exceed 99% stop rate.

## Speed benchmark

This follows PR #33602: TP4, 8K input / 1K output, 256 prompts, concurrency
4/8/16/32/64. Both arms use the same server configuration and differ only by
`SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj`.

| C | BF16 TTFT | PTPC TTFT | Delta | BF16 ITL | PTPC ITL | Delta | BF16 E2EL | PTPC E2EL | Delta | BF16 out tok/s | PTPC out tok/s | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 885.78 | 880.03 | -0.65% | 9.672 | 9.659 | -0.14% | 10780.72 | 10761.14 | -0.18% | 379.89 | 380.58 | +0.18% |
| 8 | 1440.84 | 1429.89 | -0.76% | 10.870 | 10.881 | +0.10% | 12560.92 | 12561.58 | +0.01% | 652.08 | 652.04 | -0.01% |
| 16 | 2379.08 | 2363.05 | -0.67% | 13.424 | 13.447 | +0.17% | 16111.47 | 16118.97 | +0.05% | 1016.70 | 1016.22 | -0.05% |
| 32 | 4143.43 | 4111.65 | -0.77% | 16.721 | 16.695 | -0.16% | 21249.20 | 21190.19 | -0.28% | 1541.63 | 1545.92 | +0.28% |
| 64 | 7633.61 | 7560.44 | -0.96% | 23.754 | 23.761 | +0.03% | 31934.38 | 31867.44 | -0.21% | 2051.23 | 2055.72 | +0.22% |

The optimization consistently reduces TTFT by 0.65-0.96%. Decode ITL and
output throughput remain within ±0.28%, as expected because decode stays BF16.

## Graphs-off attribution

Paired TP4 8K/16 concurrency-64 stage profiles were captured for attribution.
The QKVParallelLinear subtotal falls from 64.70 ms to 48.12 ms across the same
four-forward trace window, a 25.6% reduction. The PTPC trace contains 102 PTPC
GEMMs: three active forwards × 34 KDA layers. The remaining below-threshold
forward uses BF16. No PTPC GEMM appears in DECODE.

The trace also contains unrelated profiler/HtoD correlation distortion, so
graphs-on serving remains the production result.

## Memory

Model-load memory rises from 75.31 GB to 76.76 GB per TP4 rank (+1.45 GB).
With the same `mem-fraction-static=0.85`, the token pool changes from 7,140,096
to 7,077,568 tokens (-0.88%). The tested maximum concurrency 64 remains
available.

## Reproduction

Component benchmark:

```bash
python3 benchmark/bench_linear_attention/bench_glm53_kda_ptpc.py \
  --tp 4 --checkpoint "$CHECKPOINT" --inner-iters 10 \
  --output "$OUTPUT"
```

Serving benchmark:

```bash
python3 -m sglang.benchmark.serving --backend sglang --port 30000 \
  --dataset-name random --random-input-len 8192 \
  --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency "$CONCURRENCY" --num-prompts 256 --seed 42 \
  --output-file "$OUTPUT"
```
