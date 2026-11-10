"""TileLang FlashAttention V100 (SM70) — vendored from GooseLLM's
3rdparty/tilelang-fa-v100. Only the paged prefill path is exposed; decode stays
on sglang's Triton backend. Adapted for sglang's tilelang 0.1.8
(T.GemmWarpPolicy accessor, decorator-form jit, BaseKernelAdapter patch).
"""

from ._paged_adapter import gather_fp8_paged_kv, grouped_decode_forward, paged_forward

__all__ = ["gather_fp8_paged_kv", "grouped_decode_forward", "paged_forward"]
