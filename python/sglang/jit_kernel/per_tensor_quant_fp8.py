"""Compatibility shim (RFC #29630 Phase 4) -> kernels.ops.quantization._jit_per_tensor_quant_fp8."""

from sglang.kernels.ops.quantization import _jit_per_tensor_quant_fp8 as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
