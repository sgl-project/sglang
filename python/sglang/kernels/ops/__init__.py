"""Public operator groups for the ``sglang.kernels`` namespace.

Each submodule corresponds to one operator group from RFC #29630. Importing a
group registers its :class:`~sglang.kernels.spec.KernelSpec` metadata and
exposes thin, lazily-dispatched wrapper callables.

Importing this package eagerly imports every group so the registry is fully
populated for inventory tooling. Group imports are metadata-only and do not
import ``torch`` or a kernel backend.
"""

from importlib import import_module

# Operator groups from the RFC's proposed shape. Populated groups expose
# callable wrappers today; the rest are reserved package placeholders that keep
# the namespace shape stable for later phases.
_GROUPS = (
    "activation",
    "attention",
    "communication",
    "diffusion",
    "elementwise",
    "embeddings",
    "gemm",
    "grammar",
    "kvcache",
    "layernorm",
    "mamba",
    "memory",
    "moe",
    "quantization",
    "sampling",
    "speculative",
    "lplb",
    "kv_canary",
)

for _group in _GROUPS:
    import_module(f"{__name__}.{_group}")

del import_module, _group

__all__ = list(_GROUPS)
