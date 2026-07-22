"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.activation._jit_activation."""

from sglang.kernels.ops.activation import _jit_activation as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
