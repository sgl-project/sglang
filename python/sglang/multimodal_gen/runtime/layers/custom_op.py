# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import torch.nn as nn

import sglang.multimodal_gen.runtime.platforms as platforms
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    _oot_forward_registry: ClassVar[dict[str, dict[type["CustomOp"], Callable]]] = {}

    @staticmethod
    def register_oot_forward(
        op_cls: type["CustomOp"], *, fn: Callable, platform_key: str
    ) -> None:
        """Register ``fn`` for an exact op class and behavioral dispatch key."""
        CustomOp._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self) -> None:
        super().__init__()
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

    def _defined_forward(self, method_name: str) -> Callable | None:
        """Return an implementation defined below ``CustomOp`` in the MRO."""
        for op_cls in type(self).__mro__:
            if op_cls is CustomOp:
                return None
            if method_name in op_cls.__dict__:
                return getattr(self, method_name)
        return None

    def dispatch_forward(self) -> Callable:
        platform = platforms.current_platform
        if platform.is_out_of_tree():
            # An empty key would silently skip the platform forward below and
            # dispatch everything to forward_oot instead.
            platform_key = platform.get_dispatch_key_name().strip()
            if not platform_key:
                raise ValueError(
                    "Out-of-tree diffusion platforms must return a non-empty "
                    "get_dispatch_key_name()"
                )
            forward = self._oot_forward_registry.get(platform_key, {}).get(type(self))
            if forward is not None:
                return partial(forward, self)
            if platform_key.isidentifier():
                platform_forward = self._defined_forward(f"forward_{platform_key}")
                if platform_forward is not None:
                    return platform_forward
            return self.forward_oot
        elif platform.is_cuda():
            return self.forward_cuda
        elif platform.is_hip():
            return self.forward_hip
        elif platform.is_npu():
            return self.forward_npu
        elif platform.is_xpu():
            return self.forward_xpu
        elif platform.is_musa():
            return self.forward_musa
        else:
            return self.forward_native

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
