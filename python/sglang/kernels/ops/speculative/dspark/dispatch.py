from __future__ import annotations

import torch


def inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA inputs take the fused triton kernel; CPU and NPU inputs
    take the torch reference implementation (this triton kernel is CUDA-only;
    CPU-side callers such as unit tests exercise the reference path).

    Checked via ``device.type`` rather than ``Tensor.is_cuda``: torch_npu's
    ``transfer_to_npu`` CUDA-compatibility shim (imported as a side effect of
    the NPU hardware backend) makes ``is_cuda`` return ``True`` for NPU
    tensors too, which would otherwise route NPU inputs into this CUDA-only
    triton kernel and fail at Ascend (BiShengIR) compilation time.
    """
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device.type == "cuda"
    raise AssertionError("kernel dispatch requires at least one tensor argument")
