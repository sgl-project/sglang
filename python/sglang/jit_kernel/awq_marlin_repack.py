"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization.awq_marlin_repack."""

from sglang.kernels.ops.quantization import awq_marlin_repack as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
