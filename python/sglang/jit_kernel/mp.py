"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.communication.mp."""

from sglang.kernels.ops.communication import mp as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
