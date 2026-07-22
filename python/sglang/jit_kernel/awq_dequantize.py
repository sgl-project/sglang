"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization.awq_dequantize."""

from sglang.kernels.ops.quantization import awq_dequantize as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
