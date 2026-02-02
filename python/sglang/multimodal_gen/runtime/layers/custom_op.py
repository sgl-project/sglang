# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

import importlib
import inspect
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()
hardware_path = "sglang.multimodal_gen.runtime.backend"


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    function_name = ""

    def __init__(self) -> None:
        super().__init__()
        self._forward_method = MethodType(self.dispatch_forward(), self)

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)

    def get_function(self, backend) -> Callable:
        filename = inspect.getmodulename(inspect.getmodule(self).__file__)
        module = importlib.import_module(f"{hardware_path}.{backend}.{filename}")
        return getattr(module, self.function_name)

    def raise_error(self):
        import traceback

        traceback.print_stack()
        raise NotImplementedError

    def use_forward_native(self) -> Callable:
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        try:
            return self.get_function("native")
        except:
            self.raise_error()

    def use_forward_cuda(self, *args, **kwargs) -> Callable:
        try:
            return self.get_function("cuda")
        except:
            self.raise_error()

    def two_step_choose(self, v1: str, v2: str) -> Callable:
        try:
            return self.get_function(v1)
        except:
            try:
                return self.get_function(v2)
            except:
                self.raise_error()

    def use_forward_hip(self, *args, **kwargs) -> Callable:
        # ROCm kernels follow the CUDA path by default.
        return self.two_step_choose("hip", "cuda")

    def use_forward_cpu(self, *args, **kwargs) -> Callable:
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.two_step_choose("cpu", "cuda")

    def use_forward_tpu(self, *args, **kwargs) -> Callable:
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.two_step_choose("tpu", "native")

    def use_forward_musa(self, *args, **kwargs) -> Callable:
        # XXX (MUSA): MUSA kernels follow the CUDA path by default.
        # At this stage, sgl-kernel support for MUSA is still under active
        # development, so we fall back to the PyTorch-native implementation.
        return self.two_step_choose("musa", "native")

    def use_forward_oot(self, *args, **kwargs) -> Callable:
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.two_step_choose("oot", "native")

    def use_forward_npu(self, *args, **kwargs) -> Callable:
        # By default, we assume that NPU ops are compatible with the
        # PyTorch-native implementation.
        return self.two_step_choose("npu", "native")

    def use_forward_xpu(self, *args, **kwargs) -> Callable:
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.two_step_choose("xpu", "native")

    def dispatch_forward(self) -> Callable:
        if _is_cuda:
            return self.use_forward_cuda()
        elif current_platform.is_hip():
            return self.use_forward_hip()
        elif current_platform.is_npu():
            return self.use_forward_npu()
        elif current_platform.is_xpu():
            return self.use_forward_xpu()
        elif current_platform.is_musa():
            return self.use_forward_musa()
        else:
            return self.use_forward_native()

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
