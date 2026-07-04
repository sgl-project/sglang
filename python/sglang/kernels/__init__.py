"""Unified public kernel namespace for SGLang (RFC #29630).

SGLang runtime code and tests should import callable kernels from
``sglang.kernels.ops.<group>``, e.g.::

    from sglang.kernels.ops.layernorm import rmsnorm
    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.kernels.ops.kvcache import reshape_and_cache_flash

Implementations still live in ``sglang.jit_kernel`` (JIT CUDA), the
``sgl_kernel`` wheel (AOT CUDA/C++), Triton op modules, etc. The ``ops.*``
functions are thin wrappers that forward to a chosen backend; the
:data:`~sglang.kernels.registry.registry` provides an inventory of every
backend so alternatives can be compared and selected.

Importing this package (and any ``ops.*`` group) does not import a kernel
backend (``sgl_kernel`` / ``sglang.jit_kernel``) or trigger JIT compilation:
registration is metadata-only and backends are imported lazily on first call.
This keeps the namespace usable for inventory tooling on a CPU-only box.
"""

from sglang.kernels.registry import KernelRegistry, register_kernel, registry
from sglang.kernels.selector import get_kernel, select_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
)

# Importing the operator groups populates the registry (metadata only). Kept
# after the core imports above (and guarded from isort reordering) so those
# modules are fully initialized first.
from sglang.kernels import ops  # noqa: E402  # isort: skip

__all__ = [
    "ops",
    "CapabilityRequirement",
    "FormatSignature",
    "KernelBackend",
    "KernelRegistry",
    "KernelSpec",
    "PlatformInfo",
    "get_kernel",
    "register_kernel",
    "registry",
    "select_kernel",
]
