"""Compatibility shim (RFC #29630 Phase 4).

The JIT rmsnorm/qknorm operators moved to
``sglang.kernels.ops.layernorm._jit_norm``. This module re-exports them so
existing ``sglang.jit_kernel.norm`` import paths keep working; it will be
removed in Phase 5.
"""

from sglang.kernels.ops.layernorm import _jit_norm as _impl

globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})
