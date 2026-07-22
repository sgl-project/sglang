"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization.gptq_marlin."""

from sglang.kernels.ops.quantization import gptq_marlin as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
