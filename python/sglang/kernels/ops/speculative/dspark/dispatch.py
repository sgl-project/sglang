from __future__ import annotations

import torch


def _first_tensor(*args, **kwargs) -> torch.Tensor:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
    raise AssertionError("kernel dispatch requires at least one tensor argument")


def inputs_on_cuda(*args, **kwargs) -> bool:
    """Select Triton by the first tensor; transfer_to_npu can alias is_cuda.

    Callers with a dedicated NPU implementation must select it first.
    """
    return _first_tensor(*args, **kwargs).is_cuda


def inputs_on_npu(*args, **kwargs) -> bool:
    return _first_tensor(*args, **kwargs).device.type == "npu"
