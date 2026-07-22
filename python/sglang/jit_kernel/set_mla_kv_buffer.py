"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.kvcache._jit_set_mla_kv_buffer."""

from sglang.kernels.ops.kvcache import _jit_set_mla_kv_buffer as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
