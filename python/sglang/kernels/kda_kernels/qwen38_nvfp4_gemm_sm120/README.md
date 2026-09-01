# Qwen3.x ModelOpt NVFP4 GEMM for SM120

This implementation was automatically optimized by the Humanize2 workflow
([PolyArch/humanize](https://github.com/PolyArch/humanize)) and
[Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents).
It was imported from [BBuf/KDA-Pilot PR #195](https://github.com/BBuf/KDA-Pilot/pull/195)
at commit `516c976cee824a236679adf6eb525275a0a9a120`.

The kernel was generated from Qwen3.8-27B NVFP4 W4A4 GEMMs captured on an
NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120). Follow-up validation on
the same GPU showed that its skinny-GEMM schedule also benefits the ModelOpt
NVFP4 MLP shapes from Qwen3.5-4B and Qwen3.5-9B:

- Qwen3.5-4B gate/up: K=2560, N=18432; down: K=9216, N=2560
- Qwen3.5-9B gate/up: K=4096, N=24576; down: K=12288, N=4096
- Qwen3.6/3.8-27B gate/up: K=5120, N=34816; down: K=17408, N=5120
- Qwen3.8-27B lm_head: K=5120, N=248320

Decode rows M in {1, 2, 4, 8, 9} are available through the low-level API for
the Qwen3.5 shapes. The captured Qwen3.8 shapes support M in {1, 9} decode and
M=4369 prefill.

M=1 is normal decode, M=9 is DSpark verification with block size 8, and
M=4369 is the captured 4K-prompt prefill after chat-template expansion. The
opt-in ModelOpt dispatch is guarded by `SGLANG_ENABLE_KDA_NVFP4_GEMM=1` and is
deliberately narrower than the low-level API: it enables the E2E-qualified
Qwen3.5-4B/9B MLP shapes at M in {1, 2, 4, 8} and the Qwen3.8-27B down
projection at M=9. Every other call falls back to FlashInfer.

The imported decode kernel was adapted for serving by streaming both FP4
weights and weight scales through L2. The source task's isolated-GEMM policy
persisted each layer's 5.6--11 MiB scale tensor; that made the GEMM faster in
isolation but displaced attention/SSM state and regressed the full DSpark
pipeline. On Qwen3.8-27B with an RTX PRO 6000, the serving policy preserved the
fixed-prompt output and acceptance metrics while improving three-round output
throughput from 128.35 to 129.60 token/s.

## Multi-model SM120 validation

The added Qwen3.5 shapes were validated on an NVIDIA RTX PRO 6000 Blackwell
Server Edition with `lmsysorg/sglang:dev-qwen38-27b-dflash2`. The pinned
ModelOpt checkpoints were:

- `AxionML/Qwen3.5-4B-NVFP4` at
  `4521f321dc8c46d255929203ae6d3062e51d52fa`
- `AxionML/Qwen3.5-9B-NVFP4` at
  `97aef92393f126bf649f310cd40861be8dad3279`

The kernel benchmark rotates through eight distinct weights in a CUDA Graph,
alternates FlashInfer and KDA timing order across five trials, and reports the
median per-call latency. All twelve M=1/M=9 rows passed the FlashInfer
correctness gate (`rtol=1e-2`, `atol=2e-2`). The twelve added Qwen3.5
M=2/M=4/M=8 rows passed the same gate.

| Model shape | M | Gate/up speedup | Down speedup |
|---|---:|---:|---:|
| Qwen3.5-4B | 1 | 1.049x | 2.027x |
| Qwen3.5-4B | 2 | 1.078x | 2.024x |
| Qwen3.5-4B | 4 | 1.095x | 2.028x |
| Qwen3.5-4B | 8 | 1.079x | 2.045x |
| Qwen3.5-4B | 9 | 1.027x | 2.028x |
| Qwen3.5-9B | 1 | 1.140x | 1.267x |
| Qwen3.5-9B | 2 | 1.154x | 1.260x |
| Qwen3.5-9B | 4 | 1.157x | 1.249x |
| Qwen3.5-9B | 8 | 1.146x | 1.251x |
| Qwen3.5-9B | 9 | 1.126x | 1.271x |
| Qwen3.6/3.8-27B | 1 | 1.027x | 1.189x |
| Qwen3.6/3.8-27B | 9 | 1.025x | 1.188x |

The geometric-mean speedup is 1.243x for the original twelve M=1/M=9 rows,
1.336x for the twelve added M=2/M=4/M=8 rows, and 1.288x across all 24 rows.

End-to-end serving used 32 fixed-seed `random-ids` requests per round, 2048
input tokens, 512 output tokens, concurrency in {1, 2, 4, 8}, and a cache flush
before every round. Each concurrency ran three baseline rounds and three KDA
rounds; one adjacent baseline round followed the candidate sweep. KDA was
enabled only through
`SGLANG_ENABLE_KDA_NVFP4_GEMM=1`.

| Model | Concurrency | Baseline tok/s | KDA tok/s | Adjacent baseline | Throughput | TPOT | E2E latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1 | 274.666 | 288.600 | 273.721 | +5.07% | +4.99% | +4.83% |
| Qwen3.5-4B | 2 | 534.445 | 555.677 | 534.087 | +3.97% | +3.98% | +3.82% |
| Qwen3.5-4B | 4 | 983.987 | 1022.559 | 985.465 | +3.92% | +3.88% | +3.77% |
| Qwen3.5-4B | 8 | 1464.628 | 1514.990 | 1464.872 | +3.44% | +3.62% | +3.29% |
| Qwen3.5-9B | 1 | 184.059 | 189.442 | 183.922 | +2.93% | +2.95% | +2.84% |
| Qwen3.5-9B | 2 | 358.935 | 368.827 | 358.679 | +2.76% | +2.82% | +2.68% |
| Qwen3.5-9B | 4 | 679.397 | 696.907 | 679.012 | +2.58% | +2.67% | +2.51% |
| Qwen3.5-9B | 8 | 1025.235 | 1050.054 | 1025.277 | +2.42% | +2.52% | +2.34% |

The candidate server logs must contain the KDA fast-path message; the E2E
runner treats a missing dispatch as a failure. The adjacent baselines reproduce
the original throughput means within 0.35%, which bounds run-order drift.
Qwen3.5-9B TTFT changes by -0.81% to +1.46% across the sweep, while its TPOT
and total E2E latency improve at every concurrency, consistent with a
decode-focused kernel.
