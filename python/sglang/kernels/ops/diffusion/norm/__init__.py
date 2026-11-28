"""Normalization kernels: RMSNorm / LayerNorm / GroupNorm and their fused epilogues.

Which implementation to pick is documented in the selection matrix in
``sglang/kernels/ops/diffusion/README.md`` -- there are several per norm type
and they differ by numerical contract (bit-exact vs close), activation layout
and backend, not by speed alone.
"""
