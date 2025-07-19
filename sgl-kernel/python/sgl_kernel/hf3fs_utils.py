from typing import List

import torch


def read_shm(shm: torch.Tensor, dst: List[torch.Tensor]) -> None:
    torch.ops.sgl_kernel.read_shm(shm, dst)


def write_shm(src: List[torch.Tensor], shm: torch.Tensor) -> None:
    torch.ops.sgl_kernel.write_shm(src, shm)
