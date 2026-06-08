# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.acceleration_policy import (
    TorchCompileAutotuneConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TorchCompileDecision:
    selected: str
    eager_ms: float | None
    compiled_ms: float | None
    reason: str


class _TorchCompileAutotuner:
    def __init__(
        self,
        eager_forward: Callable[..., Any],
        compile_kwargs: Mapping[str, Any],
        config: TorchCompileAutotuneConfig,
        module_name: str,
    ) -> None:
        self._eager_forward = eager_forward
        self._compile_kwargs = dict(compile_kwargs)
        self._config = config
        self._module_name = module_name
        self._compiled_forward: Callable[..., Any] | None = None
        self._decisions: dict[tuple, TorchCompileDecision] = {}
        self._compile_failure_logged = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._config.policy == "force_compile":
            return self._get_compiled_forward()(*args, **kwargs)
        if self._config.policy in {"off", "force_eager"}:
            return self._eager_forward(*args, **kwargs)

        key = self._cache_key(args, kwargs)
        decision = self._decisions.get(key)
        if decision is None:
            if self._config.live_miss or self._is_warmup_forward():
                decision = self._select_forward(key, args, kwargs)
            else:
                return self._eager_forward(*args, **kwargs)

        if decision.selected == "compiled":
            return self._get_compiled_forward()(*args, **kwargs)
        return self._eager_forward(*args, **kwargs)

    def _get_compiled_forward(self) -> Callable[..., Any]:
        if self._compiled_forward is None:
            if hasattr(self._eager_forward, "_torchdynamo_orig_callable"):
                self._compiled_forward = self._eager_forward
            else:
                self._compiled_forward = torch.compile(
                    self._eager_forward, **self._compile_kwargs
                )
        return self._compiled_forward

    def _is_warmup_forward(self) -> bool:
        forward_context = get_forward_context_or_none()
        if forward_context is None:
            return False
        return bool(getattr(forward_context.forward_batch, "is_warmup", False))

    def _select_forward(
        self, key: tuple, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> TorchCompileDecision:
        eager_ms = self._time_candidate(lambda: self._eager_forward(*args, **kwargs))
        try:
            compiled_forward = self._get_compiled_forward()
            compiled_ms = self._time_candidate(
                lambda: compiled_forward(*args, **kwargs)
            )
        except Exception as e:
            torch.cuda.synchronize()
            decision = TorchCompileDecision(
                selected="eager",
                eager_ms=eager_ms,
                compiled_ms=None,
                reason=f"compile_failed:{type(e).__name__}",
            )
            self._decisions[key] = decision
            if not self._compile_failure_logged:
                logger.warning(
                    "torch.compile autotune failed for %s, falling back to eager: %s",
                    self._module_name,
                    e,
                )
                self._compile_failure_logged = True
            return decision

        speedup = eager_ms / compiled_ms if compiled_ms > 0 else math.inf
        if speedup >= self._config.min_speedup:
            decision = TorchCompileDecision(
                selected="compiled",
                eager_ms=eager_ms,
                compiled_ms=compiled_ms,
                reason="compiled_fastest",
            )
        else:
            decision = TorchCompileDecision(
                selected="eager",
                eager_ms=eager_ms,
                compiled_ms=compiled_ms,
                reason="speedup_below_threshold",
            )
        self._decisions[key] = decision
        logger.info(
            "torch.compile autotune selected %s for %s (%s): eager=%.3f ms, compiled=%.3f ms, speedup=%.3fx, threshold=%.3fx",
            decision.selected,
            self._module_name,
            self._shape_summary(args, kwargs),
            eager_ms,
            compiled_ms,
            speedup,
            self._config.min_speedup,
        )
        return decision

    def _time_candidate(self, fn: Callable[[], Any]) -> float:
        result = None
        for _ in range(self._config.warmup):
            result = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self._config.iters):
            result = fn()
        end.record()
        torch.cuda.synchronize()
        del result
        return start.elapsed_time(end) / self._config.iters

    def _cache_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
        return (
            self._module_name,
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            tuple(_value_key(arg) for arg in args),
            tuple((key, _value_key(value)) for key, value in sorted(kwargs.items())),
        )

    def _shape_summary(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        values = [_value_summary(arg) for arg in args]
        values.extend(
            f"{key}={_value_summary(value)}" for key, value in sorted(kwargs.items())
        )
        values = [value for value in values if value]
        return ", ".join(values[:8])


def install_torch_compile_autotune(
    module: nn.Module,
    compile_kwargs: Mapping[str, Any],
    config: TorchCompileAutotuneConfig,
    module_name: str,
) -> bool:
    forward = module.forward
    if getattr(forward, "_sglang_torch_compile_autotune", None) is not None:
        return False

    controller = _TorchCompileAutotuner(
        eager_forward=forward,
        compile_kwargs=compile_kwargs,
        config=config,
        module_name=module_name,
    )

    @wraps(forward)
    def autotuned_forward(*args: Any, **kwargs: Any) -> Any:
        return controller(*args, **kwargs)

    autotuned_forward._sglang_torch_compile_autotune = controller
    module.forward = autotuned_forward
    return True


def _value_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_key(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_value_key(item) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _value_key(item))
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            ),
        )
    return ("object", type(value).__module__, type(value).__qualname__)


def _tensor_key(tensor: torch.Tensor) -> tuple:
    device_index = tensor.device.index
    capability = None
    if tensor.device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device_index is None else device_index
        )
        capability = torch.cuda.get_device_capability(device_index)
    return (
        "tensor",
        tensor.device.type,
        device_index,
        capability,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.requires_grad,
    )


def _value_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)}:{value.dtype}:{value.device.type}"
    if isinstance(value, (list, tuple)):
        tensor_items = [_value_summary(item) for item in value]
        tensor_items = [item for item in tensor_items if item]
        if tensor_items:
            return f"{type(value).__name__}[{', '.join(tensor_items[:4])}]"
    if isinstance(value, Mapping):
        tensor_items = [
            f"{key}:{_value_summary(item)}"
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        ]
        tensor_items = [item for item in tensor_items if not item.endswith(":")]
        if tensor_items:
            return f"dict[{', '.join(tensor_items[:4])}]"
    return ""
