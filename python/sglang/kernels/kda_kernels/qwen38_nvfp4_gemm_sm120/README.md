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
ModelOpt dispatch is deliberately narrower than the low-level API: it enables
the E2E-qualified Qwen3.5-4B/9B MLP shapes at M in {1, 2, 4, 8} and the
Qwen3.8-27B down projection at M=9. Every other call falls back to FlashInfer.

The imported decode kernel was adapted for serving by streaming both FP4
weights and weight scales through L2. The source task's isolated-GEMM policy
persisted each layer's 5.6--11 MiB scale tensor; that made the GEMM faster in
isolation but displaced attention/SSM state and regressed the full DSpark
pipeline. On Qwen3.8-27B with an RTX PRO 6000, the serving policy preserved the
fixed-prompt output and acceptance metrics while improving three-round output
throughput from 128.35 to 129.60 token/s.

## Multi-model SM120 validation

The added Qwen3.5 shapes were revalidated on an NVIDIA RTX PRO 6000 Blackwell
Server Edition with PyTorch `2.13.0+cu130` and `flashinfer-python==0.6.18`.
Stale FlashInfer 0.6.17 cubin and JIT-cache packages were removed rather than
bypassing FlashInfer's package-version guard. The pinned ModelOpt checkpoints
were:

- `AxionML/Qwen3.5-4B-NVFP4` at
  `4521f321dc8c46d255929203ae6d3062e51d52fa`
- `AxionML/Qwen3.5-9B-NVFP4` at
  `97aef92393f126bf649f310cd40861be8dad3279`

The kernel benchmark rotates through eight distinct weights in a CUDA Graph,
alternates FlashInfer and KDA timing order across five trials, and reports the
median per-call latency. All 20 Qwen3.5 rows (two models, two MLP roles, and
M in {1, 2, 4, 8, 9}) passed the FlashInfer correctness gate (`rtol=1e-2`,
`atol=2e-2`).

| Model shape | M | Gate/up speedup | Down speedup |
|---|---:|---:|---:|
| Qwen3.5-4B | 1 | 1.052x | 2.023x |
| Qwen3.5-4B | 2 | 1.031x | 2.024x |
| Qwen3.5-4B | 4 | 1.055x | 2.026x |
| Qwen3.5-4B | 8 | 1.052x | 2.047x |
| Qwen3.5-4B | 9 | 1.027x | 2.039x |
| Qwen3.5-9B | 1 | 1.135x | 1.258x |
| Qwen3.5-9B | 2 | 1.134x | 1.259x |
| Qwen3.5-9B | 4 | 1.139x | 1.251x |
| Qwen3.5-9B | 8 | 1.132x | 1.252x |
| Qwen3.5-9B | 9 | 1.124x | 1.262x |

The geometric-mean speedup is 1.319x across the 16 production-dispatched
M=1/M=2/M=4/M=8 rows and 1.318x across all 20 rows. M=9 remains available
through the low-level API but is not production-dispatched for Qwen3.5.

End-to-end serving used 32 fixed-seed `random-ids` requests per round, 2048
input tokens, 512 output tokens, concurrency in {1, 2, 4, 8}, and a cache flush
before every round. Each concurrency ran three baseline rounds and three KDA
rounds; one adjacent baseline round followed the candidate sweep.

| Model | Concurrency | Baseline tok/s | KDA tok/s | Adjacent baseline | Throughput | TPOT | E2E latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1 | 266.218 | 289.452 | 265.647 | +8.73% | +8.31% | +8.03% |
| Qwen3.5-4B | 2 | 522.936 | 558.694 | 522.736 | +6.84% | +6.85% | +6.41% |
| Qwen3.5-4B | 4 | 960.507 | 1028.929 | 962.763 | +7.12% | +7.02% | +6.65% |
| Qwen3.5-4B | 8 | 1427.207 | 1520.322 | 1427.246 | +6.52% | +6.63% | +6.10% |
| Qwen3.5-9B | 1 | 184.624 | 190.493 | 184.476 | +3.18% | +3.19% | +3.08% |
| Qwen3.5-9B | 2 | 359.919 | 369.908 | 360.482 | +2.78% | +2.82% | +2.71% |
| Qwen3.5-9B | 4 | 678.651 | 697.802 | 680.757 | +2.82% | +2.81% | +2.74% |
| Qwen3.5-9B | 8 | 1022.864 | 1050.453 | 1024.721 | +2.70% | +2.78% | +2.62% |

Every formal round completed all 32 requests (65,536 input and 16,384 output
tokens). Candidate server logs contained the KDA fast-path message. The adjacent
baselines
reproduce the original throughput means within 0.32%, which bounds run-order
drift. TTFT changes range from -1.93% to +3.57% across both models, while TPOT
and total E2E latency improve at every concurrency, consistent with a
decode-focused kernel.
