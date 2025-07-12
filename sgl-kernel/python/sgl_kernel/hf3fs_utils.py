from typing import List

import torch


def read_shm(shm: torch.Tensor, dst: List[torch.Tensor], page_bytes: int) -> None:
    torch.ops.sgl_kernel.read_shm(shm, dst, page_bytes)


def write_shm(src: List[torch.Tensor], shm: torch.Tensor, page_bytes: int) -> None:
    torch.ops.sgl_kernel.write_shm(src, shm, page_bytes)
