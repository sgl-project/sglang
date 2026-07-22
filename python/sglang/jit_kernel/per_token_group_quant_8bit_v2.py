"""Compatibility shim (RFC #29630 Phase 4) -> sglang.kernels.ops.quantization._jit_per_token_group_quant_8bit_v2."""

from sglang.kernels.ops.quantization import _jit_per_token_group_quant_8bit_v2 as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
