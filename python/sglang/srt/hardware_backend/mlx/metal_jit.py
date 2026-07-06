"""Declaration surface for JIT Metal kernels (``mx.fast.metal_kernel``).

One ``MetalJitKernel`` owns a body-only Metal source and compiles lazily per
dtype key. ``aot.py`` is the separate policy layer for precompiled kernels.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx


def dtype_tag(dtype: mx.Dtype) -> str:
    # Metal's host_name attribute rejects '.', so strip the mlx.core prefix.
    return str(dtype).replace("mlx.core.", "").replace(".", "_")


class MetalJitKernel:
    """One body-only Metal kernel source, compiled lazily per dtype key.

    ``name_template`` is formatted with ``dtype_tag``-ed keys, so migrated
    kernels keep their pre-abstraction names (profiler labels, shader cache
    keys).
    """

    def __init__(
        self,
        name_template: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
    ):
        self._name_template = name_template
        self._input_names = input_names
        self._output_names = output_names
        self._source = source
        self._cache: dict[tuple[mx.Dtype, ...], object] = {}
        self._warmed: set[tuple] = set()

    def get(self, *dtypes: mx.Dtype):
        """Compiled kernel for this dtype key, cached per instance."""
        if dtypes not in self._cache:
            name = self._name_template.format(*(dtype_tag(d) for d in dtypes))
            self._cache[dtypes] = mx.fast.metal_kernel(
                name=name,
                input_names=self._input_names,
                output_names=self._output_names,
                source=self._source,
            )
        return self._cache[dtypes]

    def warm_once(self, key: tuple, dispatch: Callable[[], mx.array]) -> bool:
        """Eval one dummy dispatch per key; True when this call did the warm.

        MLX specializes Metal source on template args at first dispatch, so
        warming at init moves the per-shape compile out of the first forward.
        """
        if key in self._warmed:
            return False
        out = dispatch()
        # A dispatch returning None would make mx.eval a silent no-op and
        # record a fake warm.
        assert isinstance(out, mx.array), "warm_once dispatch must return an mx.array"
        mx.eval(out)
        self._warmed.add(key)
        return True


# Name-keyed registry: kernel modules register their source once at import
# time and dispatch through get()/warm_once(), instead of each holding its own
# module-level MetalJitKernel plus a _get_kernel wrapper for test spies.
_REGISTRY: dict[str, MetalJitKernel] = {}


def register(
    name: str,
    *,
    name_template: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
) -> None:
    """Register one kernel under `name`, dispatched via get()/warm_once().

    `name` is the registry key (e.g. "fused_moe_combine"); `name_template` is
    the on-device Metal entry name, kept separate so shader cache keys and
    profiler labels are unaffected by the registry key chosen.
    """
    if name in _REGISTRY:
        raise ValueError(f"metal_jit: kernel {name!r} already registered")
    _REGISTRY[name] = MetalJitKernel(name_template, input_names, output_names, source)


def get(name: str, *dtypes: mx.Dtype):
    """Compiled kernel for (name, dtypes). The dispatch seam tests spy on."""
    return _REGISTRY[name].get(*dtypes)


def warm_once(name: str, key: tuple, dispatch: Callable[[], mx.array]) -> bool:
    """Warm `name`'s kernel for `key`; see MetalJitKernel.warm_once."""
    return _REGISTRY[name].warm_once(key, dispatch)
