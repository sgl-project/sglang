"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.kvcache.hicache."""

from sglang.kernels.ops.kvcache import hicache as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
