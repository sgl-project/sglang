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

Decode rows M in {1, 9} are available through the low-level API for all listed
shapes. The captured Qwen3.8 shapes additionally support M=4369 prefill.

M=1 is normal decode, M=9 is DSpark verification with block size 8, and
M=4369 is the captured 4K-prompt prefill after chat-template expansion. The
opt-in ModelOpt dispatch is guarded by `SGLANG_ENABLE_KDA_NVFP4_GEMM=1` and is
deliberately narrower than the low-level API: it enables the E2E-qualified
Qwen3.5-4B/9B MLP shapes at M=1 and the Qwen3.8-27B down projection at M=9.
Every other call falls back to FlashInfer.

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
correctness gate (`rtol=1e-2`, `atol=2e-2`).

| Model shape | M | Gate/up speedup | Down speedup |
|---|---:|---:|---:|
| Qwen3.5-4B | 1 | 1.049x | 2.027x |
| Qwen3.5-4B | 9 | 1.027x | 2.028x |
| Qwen3.5-9B | 1 | 1.140x | 1.267x |
| Qwen3.5-9B | 9 | 1.126x | 1.271x |
| Qwen3.6/3.8-27B | 1 | 1.027x | 1.189x |
| Qwen3.6/3.8-27B | 9 | 1.025x | 1.188x |

The twelve-row geometric-mean kernel speedup is 1.243x.

End-to-end serving used ten fixed-seed `random-ids` requests per round, 2048
input tokens, 512 output tokens, concurrency 1, and a cache flush before every
round. Each comparison ran three baseline rounds, three KDA rounds, then one
adjacent baseline round. KDA was enabled only through
`SGLANG_ENABLE_KDA_NVFP4_GEMM=1`.

| Model | Metric | Baseline mean | KDA mean | Adjacent baseline | Improvement |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | Output throughput (tok/s) | 271.407 | 286.381 | 270.667 | +5.52% |
| Qwen3.5-4B | Mean TPOT (ms) | 3.552 | 3.361 | 3.560 | +5.40% |
| Qwen3.5-4B | Mean TTFT (ms) | 69.396 | 68.989 | 70.720 | +0.59% |
| Qwen3.5-9B | Output throughput (tok/s) | 183.691 | 189.035 | 183.383 | +2.91% |
| Qwen3.5-9B | Mean TPOT (ms) | 5.294 | 5.139 | 5.299 | +2.93% |
| Qwen3.5-9B | Mean TTFT (ms) | 80.180 | 80.763 | 82.556 | -0.73% |

The candidate server logs must contain the KDA fast-path message; the E2E
runner treats a missing dispatch as a failure. The adjacent baselines reproduce
the original throughput means within 0.3%, which bounds run-order drift.
