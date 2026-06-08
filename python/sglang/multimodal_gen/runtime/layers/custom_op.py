# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang.kernel_api_logging import debug_kernel_api
from sglang.multimodal_gen.runtime.acceleration_policy import (
    custom_op_kernel_compile_policy,
    kernel_compile_autotune_config,
    kernel_compile_kwargs,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()
_CUSTOM_OP_KERNEL_AUTOTUNE_CACHE: dict[tuple, "KernelCompileDecision"] = {}


@dataclass(frozen=True)
class KernelCompileDecision:
    selected: str
    fused_ms: float | None
    compiled_ms: float | None
    reason: str


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._compiled_forward_native: Callable | None = None
        self._kernel_compile_failure_logged = False
        self._forward_method = self.dispatch_forward()

    @debug_kernel_api
    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs) -> Any:
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs) -> Any:
        # ROCm kernels follow the CUDA path by default.
        return self.forward_cuda(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> Any:
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> Any:
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs) -> Any:
        # MUSA kernels follow the CUDA path by default.
        return self.forward_cuda(*args, **kwargs)

    def forward_oot(self, *args, **kwargs) -> Any:
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_npu(self, *args, **kwargs) -> Any:
        # By default, we assume that NPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self) -> Callable:
        if _is_cuda:
            policy = custom_op_kernel_compile_policy(
                getattr(self, "name", None), type(self).__name__
            )
            if policy == "force_torch_compile":
                return self._get_compiled_forward_native()
            if policy == "auto":
                return self._autotuned_forward
            return self.forward_cuda
        elif current_platform.is_hip():
            return self.forward_hip
        elif current_platform.is_npu():
            return self.forward_npu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_musa():
            return self.forward_musa
        else:
            return self.forward_native

    def _autotuned_forward(self, *args, **kwargs) -> Any:
        if torch.is_grad_enabled():
            return self.forward_cuda(*args, **kwargs)

        policy = custom_op_kernel_compile_policy(
            getattr(self, "name", None), type(self).__name__
        )
        if policy != "auto":
            return self.forward_cuda(*args, **kwargs)

        key = self._kernel_autotune_key(args, kwargs)
        decision = _CUSTOM_OP_KERNEL_AUTOTUNE_CACHE.get(key)
        forward_context = get_forward_context_or_none()
        is_warmup_forward = bool(
            forward_context is not None
            and getattr(forward_context.forward_batch, "is_warmup", False)
        )
        config = kernel_compile_autotune_config()
        if decision is None:
            if config.live_miss or is_warmup_forward:
                decision = self._select_kernel_forward(
                    key, args, kwargs, commit=is_warmup_forward
                )
            else:
                return self.forward_cuda(*args, **kwargs)
        elif is_warmup_forward:
            self._commit_kernel_decision(decision, commit=True)

        if decision.selected == "compiled":
            if self._compiled_forward_native is None and not (
                config.live_miss or is_warmup_forward
            ):
                return self.forward_cuda(*args, **kwargs)
            return self._get_compiled_forward_native()(*args, **kwargs)
        return self.forward_cuda(*args, **kwargs)

    def _select_kernel_forward(
        self,
        key: tuple,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        commit: bool,
    ) -> KernelCompileDecision:
        fused_ms = self._time_kernel_candidate(
            lambda: self.forward_cuda(*args, **kwargs)
        )
        try:
            compiled_forward = self._get_compiled_forward_native()
            compiled_ms = self._time_kernel_candidate(
                lambda: compiled_forward(*args, **kwargs)
            )
            if not self._outputs_match(
                self.forward_cuda(*args, **kwargs), compiled_forward(*args, **kwargs)
            ):
                decision = KernelCompileDecision(
                    selected="fused",
                    fused_ms=fused_ms,
                    compiled_ms=compiled_ms,
                    reason="compiled_mismatch",
                )
                _CUSTOM_OP_KERNEL_AUTOTUNE_CACHE[key] = decision
                self._commit_kernel_decision(decision, commit)
                logger.warning(
                    "CustomOp autotune selected fused for %s because compiled output mismatched (%s)",
                    self._op_label(),
                    self._shape_summary(args, kwargs),
                )
                return decision
        except Exception as e:
            torch.cuda.synchronize()
            decision = KernelCompileDecision(
                selected="fused",
                fused_ms=fused_ms,
                compiled_ms=None,
                reason=f"compile_failed:{type(e).__name__}",
            )
            _CUSTOM_OP_KERNEL_AUTOTUNE_CACHE[key] = decision
            self._commit_kernel_decision(decision, commit)
            if not self._kernel_compile_failure_logged:
                logger.warning(
                    "CustomOp autotune failed for %s, falling back to fused: %s",
                    self._op_label(),
                    e,
                )
                self._kernel_compile_failure_logged = True
            return decision

        speedup = fused_ms / compiled_ms if compiled_ms > 0 else math.inf
        config = kernel_compile_autotune_config()
        decision = KernelCompileDecision(
            selected="compiled" if speedup >= config.min_speedup else "fused",
            fused_ms=fused_ms,
            compiled_ms=compiled_ms,
            reason=(
                "compiled_fastest"
                if speedup >= config.min_speedup
                else "speedup_below_threshold"
            ),
        )
        _CUSTOM_OP_KERNEL_AUTOTUNE_CACHE[key] = decision
        logger.info(
            "CustomOp autotune selected %s for %s (%s): fused=%.4f ms, compiled=%.4f ms, speedup=%.3fx, threshold=%.3fx",
            decision.selected,
            self._op_label(),
            self._shape_summary(args, kwargs),
            fused_ms,
            compiled_ms,
            speedup,
            config.min_speedup,
        )
        self._commit_kernel_decision(decision, commit)
        return decision

    def _commit_kernel_decision(
        self, decision: KernelCompileDecision, commit: bool
    ) -> None:
        config = kernel_compile_autotune_config()
        if not commit or config.live_miss:
            return
        if decision.selected == "compiled":
            self._forward_method = self._get_compiled_forward_native()
        else:
            self._forward_method = self.forward_cuda
            self._compiled_forward_native = None
        logger.debug(
            "CustomOp autotune committed %s forward for %s",
            decision.selected,
            self._op_label(),
        )

    def _time_kernel_candidate(self, fn: Callable[[], Any]) -> float:
        config = kernel_compile_autotune_config()
        result = None
        for _ in range(config.warmup):
            result = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(config.iters):
            result = fn()
        end.record()
        torch.cuda.synchronize()
        del result
        return start.elapsed_time(end) / config.iters

    def _kernel_autotune_key(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple:
        return (
            self._op_label(),
            self._kernel_autotune_state_key(),
            torch.is_inference_mode_enabled(),
            tuple(_value_key(arg) for arg in args),
            tuple((key, _value_key(value)) for key, value in sorted(kwargs.items())),
        )

    def _kernel_autotune_state_key(self) -> str:
        return self.extra_repr()

    def _op_label(self) -> str:
        return str(getattr(self, "name", type(self).__name__))

    def _shape_summary(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        values = [_value_summary(arg) for arg in args]
        values.extend(
            f"{key}={_value_summary(value)}" for key, value in sorted(kwargs.items())
        )
        values = [value for value in values if value]
        return ", ".join(values[:6])

    def _outputs_match(self, lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                return False
            if torch.is_floating_point(lhs):
                return bool(torch.allclose(lhs, rhs, rtol=1e-2, atol=1e-2))
            return bool(torch.equal(lhs, rhs))
        if isinstance(lhs, tuple) and isinstance(rhs, tuple) and len(lhs) == len(rhs):
            return all(
                self._outputs_match(l_item, r_item)
                for l_item, r_item in zip(lhs, rhs)
            )
        if isinstance(lhs, list) and isinstance(rhs, list) and len(lhs) == len(rhs):
            return all(
                self._outputs_match(l_item, r_item)
                for l_item, r_item in zip(lhs, rhs)
            )
        return lhs == rhs

    def _get_compiled_forward_native(self) -> Callable:
        if self._compiled_forward_native is None:
            logger.info_once(
                "Using torch.compile native path for custom op "
                f"{getattr(self, 'name', type(self).__name__)}"
            )
            if hasattr(self.forward_native, "_torchdynamo_orig_callable"):
                self._compiled_forward_native = self.forward_native
            else:
                self._compiled_forward_native = torch.compile(
                    self.forward_native, **kernel_compile_kwargs()
                )
        return self._compiled_forward_native

    @classmethod
    def enabled(cls) -> bool:
        # since we are not using Inductor, we always return True
        return True

    @staticmethod
    def default_on() -> bool:
        """
        On by default if level < CompilationLevel.PIECEWISE
        Specifying 'all' or 'none' in custom_op takes precedence.
        """
        raise NotImplementedError

    # Dictionary of all custom ops (classes, indexed by registered name).
    # To check if an op with a name is enabled, call .enabled() on the class.
    # Examples:
    # - MyOp.enabled()
    # - op_registry["my_op"].enabled()
    op_registry: dict[str, type["CustomOp"]] = {}

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str) -> Callable:

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator


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
