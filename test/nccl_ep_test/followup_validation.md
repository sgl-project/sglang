# NCCL EP follow-up validation

These changes build on the persistent Graph lifecycle at
`537b7a17477a6621efd727f5f088848d754ed3b4` (upstream PR #38683, dependent on #32329).

## Triton compute compatibility

Select `--moe-a2a-backend nccl_ep --moe-runner-backend triton` explicitly.
The adapter supports ordinary block-128 FP8, float32 activation scales and
gated SiLU. It preserves the LL wire contract and delegates the original
routing weights to NCCL combine. It does not need the native DeepEP library
or DeepGEMM for expert computation.

This is a compatibility path: dequantization, requantization, alignment and
scratch scale with expert receive capacity. No LL performance parity is claimed.

Local compute tests ran on RTX 4060 Laptop (SM89), Torch `2.13.0+cu130`.
They execute real Triton GEMMs and ordinary CUDA Graphs, without native EP.
CPU checks use fan-in-normalized random weights and `rtol=atol=0.02`.
The higher-gain stress case is separately checked against existing unpadded
single-expert Triton GEMMs: FP8 GEMM rounding amplified by the second
quantization does not satisfy that CPU tolerance for every high-gain input.
Eager/Graph valid-row comparisons are exact. Invalid payload and scale tails
are poisoned with NaN to verify masking.

Run from the checkout root:

```bash
PYTHONPATH=python:test python -m nccl_ep_test.single_gpu --report /tmp/nccl-ep-compute.json
```

Before two-GPU integration, run the same gate on one PRO 6000, adding
`--require-sm 120 --experts 32 --capacity 64`. This records actual peak memory
and replay time. SM120, real EP and model serving remain pending until their
respective hardware gates pass. The former server's Torch 2.11 environment
and the local Torch 2.13 environment are separate validation records.
