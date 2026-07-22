"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm."""

from sglang.kernels.ops.gemm import cutedsl_dsv3_fused_a_gemm as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
