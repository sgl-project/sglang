"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.communication.all_reduce."""

from sglang.kernels.ops.communication import all_reduce as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
