"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.kvcache.triton_store_cache."""

from sglang.kernels.ops.kvcache import triton_store_cache as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
