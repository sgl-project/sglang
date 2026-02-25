from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_stream_wait_value_module() -> Module:
    return load_jit(
        "cuda_wait_value",
        cuda_files=["cuda_wait_value.cuh"],
        cuda_wrappers=[("stream_wait_value", "stream_wait_value")],
    )


def stream_wait_value(flag: torch.Tensor, value: int) -> None:
    module = _jit_stream_wait_value_module()
    module.stream_wait_value(flag, value)


class Event:
    def __init__(self) -> None:
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")

    def record(self, value: int = 1) -> None:
        self.flag[0] = value

    def wait(self, value: int = 1) -> None:
        stream_wait_value(self.flag, value)
