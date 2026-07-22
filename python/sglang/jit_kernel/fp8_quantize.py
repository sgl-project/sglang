"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization.fp8_quantize."""

from sglang.kernels.ops.quantization import fp8_quantize as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
