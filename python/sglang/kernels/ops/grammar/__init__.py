"""Constrained-decoding / grammar kernels (Triton).

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.grammar.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# (module, public_fn) migrated from constrained/triton_ops.
_TRITON_KERNELS = [
    ("bitmask_ops", "apply_token_bitmask_inplace_triton"),
    ("token_filter_ops", "set_token_filter_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"grammar.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.grammar.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__ = []
