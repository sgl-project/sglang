"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.mamba.inkling_sconv."""

from sglang.kernels.ops.mamba import inkling_sconv as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
