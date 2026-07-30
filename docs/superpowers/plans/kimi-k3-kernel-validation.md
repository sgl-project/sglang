# Kimi-K3 Standalone Kernel Validation

Date: 2026-07-30

Source revision: `578edb240a6d6f6f2fa4c31497276955d7f73432`

Base revision: `f4e0ac382e4e5d644f2fbe4a15c20da53500bbca`

## Scope and Static Checks

- `python/sglang/kernels/**` is byte-for-byte identical to the locked source
  revision according to `git diff HEAD 578edb240a -- python/sglang/kernels`.
- No file under `benchmark/` is added or modified.
- Outside `python/sglang/kernels`, implementation changes are limited to the
  generic CustomAllReduceV2 storage-plane support and four environment entries.
- `pre-commit run --all-files` passed.
- `git diff --check` passed.
- `python3 -m compileall -q python/sglang/kernels test/registered/kernels
  test/registered/jit` passed.
- The registered-test CI registry validator passed.

## H100 Validation

The available Radix assignment `host-85-234-79-221` provided 8 NVIDIA H100
80GB GPUs with PyTorch `2.11.0+cu130`. Tests used the exact branch snapshot in
an isolated repository at `/data/bbuf/repos/sglang-k3-kernels`.

| Test group | Result |
|---|---:|
| fused-decode contract, add3, top-p renorm, tiny GEMM | 284 passed |
| KDA ReplaySSM fold/ring tests | 34 passed |
| vision RoPE and generic MoE kernels | 185 passed, 1 skipped |
| existing KDA parity matrix, including packed-decode C++ fast path | 14 passed |
| image normalize-and-patchify | 2 passed |

Total supported H100 coverage: **519 passed**. Pytest also reported the
parameter-loop subtests separately; they are not added to this total.

The KDA parity matrix exercised large batches, grouped heads, padding slots,
and the C++ packed-decode dispatch against the Triton/reference path.

## Architecture-Gated Coverage

The fused all-reduce test was also launched on four H100 GPUs. The host CUDA
driver declined symmetric-memory multicast initialization before any K3
collective kernel ran. The test now applies one module-level SM100 gate, so the
same unsupported run exits cleanly as `50 skipped` in 0.19 seconds instead of
repeating communicator initialization for every parameter case. On its
registered B200 runner, failure to obtain multicast remains a hard error.

The following direct tests are registered but were not run locally because no
SM100/SM103 assignment was reachable:

- `test_kda_nvidia_prefill.py` — B200 CuTe prefill versus vendored FLA.
- `test_kda_ptx_prefill.py` — GB300 PTX/tcgen05 prefill versus vendored FLA.
- `test_kda_mtp_cutedsl_replayssm_ring.py`.
- K3 SiTU, MLA gate, and attention-residual TMA tests.
- K3 fused all-reduce, GEMM+AG, GEMM+AR, SP collectives, and fused
  attention-residual collective tests.

Hardware attempts were kept read-only until access was confirmed:

- The configured Verda B300 assignment had expired; the current Radix
  assignment resolved to H100 hardware.
- The Cirrascale B200 host rejected the configured public key.
- The standalone B200 host closed the SSH connection.
- The Ion B200 host rejected authentication.

No result from H100 is presented as SM100 validation.

## Required Blackwell Follow-Up

Run these registered files on their declared CI hardware:

```text
base-b-kernel-unit-test-4-gpu-b200
  test_kda_nvidia_prefill.py
  test_kda_mtp_cutedsl_replayssm_ring.py
  test_attn_res_fused_tma.py
  test_situ_and_mul.py
  test_situ_mul_quant.py
  test_mla_output_gate.py
  test_ar_fusion.py
  test_gemm_ag.py
  test_k3_gemm_ar.py
  test_sp_collective.py

base-c-test-4-gpu-gb300
  test_kda_ptx_prefill.py
```

Passing these jobs completes the architecture-specific acceptance matrix.
