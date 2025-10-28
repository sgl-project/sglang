# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from sglang.multimodal_gen.runtime.utils.common import (
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_is_xpu = is_xpu()


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self) -> None:
        super().__init__()
        self._forward_method = self.dispatch_forward()

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

    def forward_cpu(self, *args, **kwargs) -> Any:
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> Any:
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs) -> Any:
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self) -> Callable:
        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_npu:
            return self.forward_npu
        elif _is_xpu:
            return self.forward_xpu
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
