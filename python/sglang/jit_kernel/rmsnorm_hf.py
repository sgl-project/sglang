"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.layernorm.rmsnorm_hf."""

from sglang.kernels.ops.layernorm import rmsnorm_hf as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
