"""SM100 CuTeDSL masked grouped GEMM, ported from the frozen base-GEMM study.

Source: branch `codex/bf16-moe-base-gemm` @ 55fae8bce7,
`benchmark/kernels/lora_moe/base_gemm/cutedsl_masked/` — the study's winning
family (swap_ab + direct schedule, canonical [E, N, K] weights). The kernel
retains its NVIDIA BSD-3 header; the bench harness tail was dropped in the
move. Port scope, runtime contract, ABI bounds, and the ordered task list are
in execution plan section 45; the provider class, compile cache, and schedule
builder land as the next tasks of that list.

Reachable from serving through `CuteDslBf16Provider` when
`SGLANG_LORA_MOE_BASE_PROVIDER=cutedsl` selects it (SM100+ only); the default
provider stays DeepGEMM until the gate-2 ruling flips it.
"""
