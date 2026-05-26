# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch hooks for layerwise NVTX profiling in SGLang Diffusion.

Mirrors the structure of ``sglang.srt.utils.nvtx_pytorch_hooks.PytHooks``
but uses a compact ``{name} in={shapes}`` marker format that is well-suited
to DiT transformer blocks. See
``sglang.srt.utils.nvtx_pytorch_hooks`` for the LLM-runtime equivalent
that emits a richer per-layer parameter dict.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.hooks import RemovableHandle

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Module types that are too lightweight to warrant their own NVTX range.
# Skipping them keeps the captured timeline readable.
_DEFAULT_SKIP_TYPES: tuple[type, ...] = (
    torch.nn.Identity,
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
)


@contextlib.contextmanager
def maybe_nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    """Context manager that wraps a block of work in an NVTX range.

    Calls ``range_push`` / ``range_pop`` directly rather than going through
    :func:`torch.cuda.nvtx.range`, which would otherwise interpret ``name`` as a
    ``str.format`` template (so a literal ``{`` in the marker would raise
    ``KeyError``). The ``range_pop`` is invoked from the ``finally`` clause, so
    exceptions raised inside the ``with`` block cannot leak a half-open range.

    When ``enabled`` is ``False`` the function is a zero-cost no-op, suitable
    for use under a per-request gate (e.g. warmup exclusion).
    """
    if not enabled:
        yield
        return
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


class DiffusionNvtxHooks:
    """Register NVTX markers around each submodule forward pass.

    Each registered module emits an NVTX range covering its forward pass.
    The range name encodes the qualified module name and the input tensor
    shapes for downstream identification in Nsight Systems.

    Hook handles are retained so they can be removed via :meth:`remove_hooks`;
    the same instance must not be reused across multiple model instances.
    """

    def __init__(self, skip_types: tuple[type, ...] = _DEFAULT_SKIP_TYPES) -> None:
        self._skip_types = skip_types
        self._module_to_name_map: dict[torch.nn.Module, str] = {}
        self._hook_handles: list[RemovableHandle] = []
        # Caller must explicitly enable via ``set_enabled``. Default off
        # so a forward that bypasses the component-use gate (e.g. an early
        # warmup pass) cannot accidentally pollute the captured timeline.
        self._enabled: bool = False

    def register_hooks(
        self,
        model: torch.nn.Module,
        prefix: str = "",
    ) -> int:
        """Walk ``model`` and attach forward pre/post hooks to every module.

        Args:
            model: Root module to instrument.
            prefix: Optional name prefix prepended to every emitted range.

        Returns:
            Number of modules instrumented.

        Notes:
            Weight-tied or otherwise duplicated module instances are
            skipped (the first occurrence wins) so each forward pass
            produces exactly one NVTX range.
        """
        instrumented = 0
        for name, module in model.named_modules(prefix=prefix):
            if isinstance(module, self._skip_types):
                continue
            # Skip duplicate module instances (e.g., weight-tied layers).
            # The check must happen before hook registration to avoid
            # double-emitting NVTX ranges on the second occurrence.
            if module in self._module_to_name_map:
                logger.debug(
                    "NVTX: module %s already registered as '%s', skipping '%s'",
                    type(module).__name__,
                    self._module_to_name_map[module],
                    name,
                )
                continue
            self._module_to_name_map[module] = name
            self._hook_handles.append(
                module.register_forward_pre_hook(
                    self._forward_pre_hook, with_kwargs=True
                )
            )
            # ``always_call=True`` (PyTorch 2.0+) guarantees the post-hook
            # still fires when ``forward`` raises, so an OOM or assertion
            # inside the wrapped module cannot leak a half-open NVTX range.
            self._hook_handles.append(
                module.register_forward_hook(self._forward_hook, always_call=True)
            )
            instrumented += 1
        return instrumented

    def remove_hooks(self) -> None:
        """Remove every hook registered by this instance.

        Safe to call multiple times; subsequent calls are no-ops. The
        bookkeeping is cleared in a ``finally`` so a misbehaving
        ``handle.remove()`` cannot leave the instance with stale
        handles or name-map entries.
        """
        try:
            for handle in self._hook_handles:
                handle.remove()
        finally:
            self._hook_handles.clear()
            self._module_to_name_map.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Toggle whether the registered hooks emit NVTX ranges.

        When disabled, both the pre- and post-hooks early-return, so each
        forward produces a matched (push, pop) pair of "no-ops" — no range
        leak and no half-open range across the toggle.
        """
        self._enabled = enabled

    # ------------------------------------------------------------------ hooks

    def _forward_pre_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not self._enabled:
            return
        name = self._module_to_name_map.get(module, "unknown")
        shapes = _collect_input_shapes(args, kwargs)
        marker = f"{name} in={shapes}" if shapes else name
        nvtx.range_push(marker)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        _args: Any,
        _output: Any,
    ) -> None:
        if not self._enabled:
            return
        nvtx.range_pop()


def _collect_input_shapes(
    args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
) -> list[list[int]]:
    """Best-effort extraction of input tensor shapes for marker labels.

    Walks positional ``args`` and keyword ``kwargs`` values, recursing into
    lists and tuples (so DiT inputs like ``image_rotary_emb=(cos, sin)`` are
    captured). Non-tensor scalars, ``None``, dicts, and arbitrary objects are
    silently skipped.
    """
    shapes: list[list[int]] = []
    _append_tensor_shapes(args, shapes)
    if kwargs:
        _append_tensor_shapes(tuple(kwargs.values()), shapes)
    return shapes


def _append_tensor_shapes(items: Any, shapes: list[list[int]]) -> None:
    if isinstance(items, torch.Tensor):
        shapes.append(list(items.size()))
        return
    if isinstance(items, (list, tuple)):
        for item in items:
            _append_tensor_shapes(item, shapes)
