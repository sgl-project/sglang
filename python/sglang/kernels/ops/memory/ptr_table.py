"""Device-pointer tables for kernels that address several tensors per launch."""

from __future__ import annotations

from typing import Sequence, Union

import torch


def make_ptr_table(
    rows: Union[Sequence[int], Sequence[Sequence[int]]],
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Pack ``data_ptr()`` values -- flat, or 2-D with companion columns such as
    strides -- into an ``int64`` table a kernel bitcasts back to pointers.

    Build unsigned, never ``dtype=torch.int64``: XPU USM addresses set the top
    bit, which ``torch.tensor()`` rejects while unpacking through ``long long``.
    ``view`` moves no bits, so kernels keep their signed element type. Values
    must be in ``[0, 2**64)``.
    """
    return torch.tensor(rows, dtype=torch.uint64, device=device).view(torch.int64)
