"""SM100 CuTeDSL masked grouped GEMM, ported from the frozen base-GEMM study.

This is the study's winning family (swap_ab + direct schedule, canonical
[E, N, K] weights). The kernel retains its NVIDIA BSD-3 header; benchmark-only
code is not part of the serving package.

The serving config reaches it through ``CuteDslBf16Provider`` on supported
devices.
"""
