# NCCL EP configuration review follow-up

This follow-up addresses configuration findings reported on #38683, #38886,
#38887 and #38888 on 2026-09-11. Those PRs share a dependency chain, so repeated
findings are fixed in the Graph parent and inherited by the later branches.
It does not migrate the stack to a different upstream baseline.

## Findings and scope

| Finding | Resolution | Validation needed |
| --- | --- | --- |
| Default EP remains 1 while the dispatcher uses the TP communicator | Include NCCL EP in TP-spanning EP resolution, including fallback resolution | Local public configuration regression; two-rank serving with `--ep` omitted |
| Default prefill exceeds the LL dispatch budget | Limit the per-DP-rank chunk to the LL budget and KV page boundary; reject unchunked prefill and PP dynamic chunking | Local real PrefillAdder long-prompt/mixed-decode regression; two-rank long-prefill eager/Graph serving |
| Unavailable NCCL EP falls back to DeepEP with an incompatible Triton runner | Keep Triton and select standard dispatch (`none`) | Local unavailable-device regression with DeepEP importable |
| Unquantized models reach an unsupported LL compute path | Reject unsupported quantization at MoE implementation selection | Local rejection and FP8/W4A-FP8/non-NCCL-EP controls |
| SM100 availability disagrees with the dispatcher scale guard | Use the same UE8M0 incompatibility reason in both gates | Local capability/scale-mode controls; this does not enable UE8M0 |
| Eager FP16 payload is interpreted as BF16 | Reject non-BF16 parameters before group creation and inputs before handle creation | Local real dispatcher with fake external EP, plus BF16 lifecycle regression |
| Non-Triton eager mode allows unsupported SBO/TBO | Reject overlap for every NCCL EP runner | Local public configuration regression |

The non-FP8 diagnosis must preserve the existing W4A-FP8 compute path. The
Blackwell diagnosis must distinguish SM120 float32 scales from the
`DEEPGEMM_BLACKWELL` UE8M0 mode. DeepSeek's `ep_size` accesses cited in the review
belong to staged overlap operations; serial forward uses `moe_ep_size`.
Rejecting unsupported overlap is intentional; this patch does not add SBO/TBO.

## Local evidence

The initial public-contract reproducer failed 16 cases and passed one on the
unmodified Graph parent (`537b7a1`). The corrected regression file contains 27
passing cases, including a 257-token request processed through the real
PrefillAdder with and without mixed decode tokens. External EP bindings and GPU
capability queries are replaced; configuration resolution, scheduling, dispatcher
guards and ordinary CUDA Graph execution remain real.

On RTX 4060 Laptop (SM89), Torch 2.13.0+cu130:

- Graph-parent NCCL EP suite: 115 passed.
- Configuration resolution/migration/namespace suite: 195 passed and 21 subtests.
- Standalone new CI entrypoint with optional NCCL EP imports blocked: 27 passed.
- Changed-file pre-commit, Mintlify build validation and internal-link checks passed.

```bash
PYTHONPATH=python:test python3 -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_*.py
```

The later branches add their own Triton, zero-token and shared-expert tests.
Their final counts are recorded in the respective PR descriptions.

## Hardware and performance limits

These fixes do not change communication kernels or persistent Graph ownership.
No new native NCCL EP execution, model benchmark or Nsight run is claimed for
this review follow-up. Earlier hardware records remain attached to their tested
SHAs; they used explicit EP sizes and small prefill chunks and do not prove the
new default-configuration behavior.

Before certifying the new serving defaults, rerun the native pair and model
eager/Graph gates on the final stack SHA. Omit `--ep`, retain TP=DP=2, use the
default chunk size with an explicit LL budget of 64, and send a prompt longer
than 64 tokens. Check the resolved EP size, actual per-rank prefill size, finite
outputs, zero-token participation and eager/Graph recovery. Reducing prefill
chunks can change throughput, so prior short-request timings do not describe
long-prefill performance. No new profile is required merely to validate rejection
of an unsupported dtype, scale mode, quantization or overlap option.

Upstream `run-ci` authorization, maintainer approval and integration with the
unmerged parent baseline remain separate from local correctness results.
