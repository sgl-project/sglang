# Qwen3.8 NVFP4 GEMM for SM120

This implementation was automatically optimized by the Humanize2 workflow
([PolyArch/humanize](https://github.com/PolyArch/humanize)) and
[Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents).
It was imported from [BBuf/KDA-Pilot PR #195](https://github.com/BBuf/KDA-Pilot/pull/195)
at commit `516c976cee824a236679adf6eb525275a0a9a120`.

The kernel is specialized for the Qwen3.8-27B NVFP4 W4A4 GEMMs captured on an
NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120):

- gate/up: K=5120, N=34816
- down: K=17408, N=5120
- lm_head: K=5120, N=248320
- serving rows: M in {1, 9, 4369}

M=1 is normal decode, M=9 is DSpark verification with block size 8, and
M=4369 is the captured 4K-prompt prefill after chat-template expansion. The
low-level API supports all captured shapes. The opt-in ModelOpt dispatch is
guarded by `SGLANG_ENABLE_KDA_NVFP4_GEMM=1` and is deliberately narrower: only
the E2E-validated down projection `(M, K, N) = (9, 17408, 5120)` uses KDA, and
every other call falls back to FlashInfer.

The imported decode kernel was adapted for serving by streaming both FP4
weights and weight scales through L2. The source task's isolated-GEMM policy
persisted each layer's 5.6--11 MiB scale tensor; that made the GEMM faster in
isolation but displaced attention/SSM state and regressed the full DSpark
pipeline. On Qwen3.8-27B with an RTX PRO 6000, the serving policy preserved the
fixed-prompt output and acceptance metrics while improving three-round output
throughput from 128.35 to 129.60 token/s.
