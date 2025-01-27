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


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)
