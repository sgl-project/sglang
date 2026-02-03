import torch

from sglang.multimodal_gen.runtime.layers.triton_ops import fuse_scale_shift_kernel


def mul_add(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0):
    return fuse_scale_shift_kernel(a, b, c, scale_constant=k)
