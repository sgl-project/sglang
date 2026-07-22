"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization.minimax_quant_ue8m0."""

from sglang.kernels.ops.quantization import minimax_quant_ue8m0 as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
