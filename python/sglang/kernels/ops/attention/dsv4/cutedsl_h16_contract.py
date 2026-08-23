"""Lightweight shared shape limits for the DSV4 CuTe DSL H16 path.

Keep this module free of Torch, CUDA, and CuTe imports so runtime validation
can use the same contract as the lazily imported kernel.
"""

DSV4_CUTEDSL_H16_TOPK_ALIGNMENT = 128
DSV4_CUTEDSL_H16_MAX_TOPK = 8192
