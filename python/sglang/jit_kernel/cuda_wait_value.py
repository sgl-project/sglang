from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@lru_cache(maxsize=1)
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


def main():
    import time

    event = Event()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    with torch.cuda.stream(stream_a):
        print("Stream A: waiting event to be recorded")
        event.wait()

    print("Stream B: waiting 5 seconds before recording event")
    with torch.cuda.stream(stream_b):
        for _ in range(5):
            time.sleep(1)
            print(".", end="", flush=True)
        print("\nStream B: recording event")
        event.record()

    stream_a.synchronize()
    print("Stream A: event recorded, proceeding")


if __name__ == "__main__":
    main()
