"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.layernorm._jit_norm."""

from sglang.kernels.ops.layernorm import _jit_norm as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
