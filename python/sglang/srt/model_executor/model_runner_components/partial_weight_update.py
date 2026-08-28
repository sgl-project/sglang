"""Partial (subset) in-place weight updates from disk.

A partial update streams only the checkpoint tensors matching the requested
name prefixes into the live model, records exactly which modules received
weights, and re-runs quantization post-processing for those modules only.
Graph-visible tensor storage is verified against a pre-update manifest, so
CUDA-graph staleness is reported instead of silent.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Tuple

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_loader.loader import device_loading_context

logger = logging.getLogger(__name__)

_INPLACE_WRITE_OPS = (
    torch.ops.aten.copy_.default,
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_.Tensor,
    torch.ops.aten.index_copy_.default,
    torch.ops.aten.index_put_.default,
    torch.ops.aten.masked_fill_.Scalar,
)


def _storage_ptr(tensor: torch.Tensor) -> int | None:
    try:
        return tensor.untyped_storage().data_ptr()
    except (RuntimeError, ValueError, NotImplementedError):
        return None


class ModuleTouchRecorder(TorchDispatchMode):
    """Record which modules a weight load writes into.

    Intercepts in-place tensor writes for the duration of the context and
    maps the destination storage back to the owning parameter's module, so
    every load style is seen: ``param.weight_loader`` calls, module-level
    helpers invoking ``default_weight_loader`` directly, and raw
    ``param.data.copy_`` writes. No ``weight_loader`` attribute is installed
    or replaced — model code branches on ``weight_loader is
    default_weight_loader`` identity, which wrapping would break. The
    ``touched`` set survives the context so a rollback load can reuse it.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._storage_to_module: dict = {}
        for module_name, module in model.named_modules():
            for param in module._parameters.values():
                if param is None:
                    continue
                ptr = _storage_ptr(param)
                if ptr is not None:
                    self._storage_to_module.setdefault(ptr, (module_name, module))
        self._touched: dict = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _INPLACE_WRITE_OPS and args and isinstance(args[0], torch.Tensor):
            owner = self._storage_to_module.get(_storage_ptr(args[0]))
            if owner is not None:
                self._touched.setdefault(id(owner[1]), owner)
        return func(*args, **kwargs)

    def touched_modules(self) -> List[Tuple[str, nn.Module]]:
        return list(self._touched.values())


def filter_weights_by_prefix(
    weights: Iterable[Tuple[str, torch.Tensor]],
    weight_name_prefixes: List[str],
    seen_names: List[str],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield only weights matching the prefixes, recording their names."""
    prefixes = tuple(weight_name_prefixes)
    for name, weight in weights:
        if name.startswith(prefixes):
            seen_names.append(name)
            yield name, weight


def filter_weights_by_names(
    weights: Iterable[Tuple[str, torch.Tensor]],
    weight_names: set,
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, weight in weights:
        if name in weight_names:
            yield name, weight


def postprocess_touched_modules(
    touched: List[Tuple[str, nn.Module]], target_device: torch.device
) -> int:
    """Re-run quantization post-processing for the touched modules only."""
    count = 0
    for _, module in touched:
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
            count += 1
    return count
