"""Compatibility shim (RFC #29630 Phase 4).

JIT activation operators moved to ``sglang.kernels.ops.activation._jit_activation``.
Re-exported here so existing ``sglang.jit_kernel.activation`` imports keep
working; removed in Phase 5.
"""

from sglang.kernels.ops.activation import _jit_activation as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
