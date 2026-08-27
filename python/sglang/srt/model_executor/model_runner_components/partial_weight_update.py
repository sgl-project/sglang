"""Partial (subset) in-place weight updates from disk.

A partial update streams only the checkpoint tensors matching the requested
name prefixes into the live model, records exactly which modules received
weights, and re-runs quantization post-processing for those modules only.
Graph-visible tensor storage is verified against a pre-update manifest, so
CUDA-graph staleness is reported instead of silent.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator, List, Tuple

import torch
from torch import nn

from sglang.srt.model_loader.loader import device_loading_context
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


class ModuleTouchRecorder:
    """Record which modules receive weights during a load.

    Weight loaders write through ``param.data``, which tensor version counters
    do not observe, so the recorder temporarily wraps every parameter's weight
    loader for the duration of the context; the model's own load path then
    reveals exactly which modules were written, including fused and expert
    parameters that name-based mapping cannot resolve generically. The
    ``touched`` set survives the context so a rollback load can reuse it.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._restores: List[Callable[[], None]] = []
        self._touched: dict = {}

    def __enter__(self) -> "ModuleTouchRecorder":
        seen: set = set()
        for module_name, module in self._model.named_modules():
            for param in module._parameters.values():
                if param is None or id(param) in seen:
                    continue
                seen.add(id(param))
                self._wrap_param(param, module_name=module_name, module=module)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for restore in self._restores:
            restore()
        self._restores.clear()
        return False

    def touched_modules(self) -> List[Tuple[str, nn.Module]]:
        return list(self._touched.values())

    def _make_recording_loader(
        self, original: Callable, module_name: str, module: nn.Module
    ) -> Callable:
        module_id = id(module)

        def recording_loader(*args, **kwargs):
            self._touched.setdefault(module_id, (module_name, module))
            return original(*args, **kwargs)

        return recording_loader

    def _wrap_param(
        self, param: nn.Parameter, *, module_name: str, module: nn.Module
    ) -> None:
        if isinstance(getattr(type(param), "weight_loader", None), property):
            # BasevLLMParameter-style: the property reads self._weight_loader.
            original = param._weight_loader
            param._weight_loader = self._make_recording_loader(
                original, module_name, module
            )
            self._restores.append(
                lambda p=param, o=original: setattr(p, "_weight_loader", o)
            )
        else:
            had_instance_attr = "weight_loader" in param.__dict__
            saved = param.__dict__.get("weight_loader")
            original = getattr(param, "weight_loader", default_weight_loader)
            param.weight_loader = self._make_recording_loader(
                original, module_name, module
            )
            if had_instance_attr:
                self._restores.append(
                    lambda p=param, o=saved: setattr(p, "weight_loader", o)
                )
            else:
                self._restores.append(lambda p=param: delattr(p, "weight_loader"))


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
