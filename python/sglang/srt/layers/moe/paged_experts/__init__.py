"""Paged Experts: serve an MoE model larger than VRAM by keeping K of E experts resident on the GPU
and paging the rest from pinned host RAM over PCIe, on demand per decode step (PagedAttention, but for
MoE experts — compute stays on the GPU; only expert *storage* is host-backed and demand-paged).

This package is the in-tree home of the out-of-tree paging engine (see the contribution plan). It wraps
the real fused-MoE quant method with a K-slot resident expert table; weight movement reuses sglang's
existing host<->device transfer kernels (``transfer_kv_*_mla``), so no custom CUDA is required for the
eager or static-wave-captured paths.
"""

from sglang.srt.layers.moe.paged_experts.guard import check_paged_experts_compat
from sglang.srt.layers.moe.paged_experts.method import make_for_layer

__all__ = ["check_paged_experts_compat", "make_for_layer"]
