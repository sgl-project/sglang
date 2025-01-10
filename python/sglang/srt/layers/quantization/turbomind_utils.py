# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from sglang.srt.utils import get_device_capability


def get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> List[torch.Tensor]:
    assert x.dtype == torch.int32
    xs = []
    for _ in range(8):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_awq_gemm(x: torch.Tensor) -> torch.Tensor:
    xs = get_u4_slices(x, torch.uint8)
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    ys = [xs[i] for i in order]
    return torch.stack(ys, dim=-1).view(*x.shape[:-1], -1)


def process_awq_gemm(x: torch.Tensor, kind: str):
    if x.dtype == torch.int32:
        x = unpack_awq_gemm(x)
    if kind in ["qweight", "qzeros", "scales"]:
        x = x.t()
    return x


def process_gptq(x: torch.Tensor, kind: str):
    if x.dtype == torch.int32:
        xs = get_u4_slices(x, torch.uint8)
        if kind == "qweight":  # (k/8,n)
            x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        else:  # 'qzeros' (k/g,n/8)
            x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
    if kind in ["qweight", "qzeros", "scales"]:
        x = x.t()
    return x


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def verify_turbomind_supported(quant_bit: int, group_size: int) -> bool:

    if quant_bit not in [4]:
        raise NotImplementedError(
            f"[Tubomind] Only 4-bit is supported for now, but got {quant_bit} bit"
        )
    if group_size != 128:
        raise NotImplementedError(
            f"[Tubomind] Only group_size 128 is supported for now, "
            f"but got group_size {group_size}"
        )

    major, minor = get_device_capability()
    capability = major * 10 + minor
    if capability < 70:
        raise NotImplementedError(
            f"[Tubomind] Only capability >= 70 is supported for now, but got {capability}"
        )

    return True
