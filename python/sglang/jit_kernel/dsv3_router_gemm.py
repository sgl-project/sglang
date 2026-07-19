"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.gemm._jit_dsv3_router_gemm."""

from sglang.kernels.ops.gemm import _jit_dsv3_router_gemm as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
