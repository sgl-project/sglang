from typing import Iterable

import torch


def set_kv_buffer_kernel(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fallback: bool = False,
):
    try:
        if fallback:
            raise RuntimeError("Fallback to torch implementation")
        torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, loc, k, v)
    except RuntimeError:  # ok, fallback to torch implementation
        k_cache[loc] = k
        v_cache[loc] = v


def allocate_pin_memory(
    shape: Iterable[int],
    dtype: torch.dtype,
    write_combined: bool = False,
    numa_affinity: int | None = None,
) -> torch.Tensor:
    shape_tuple = tuple(shape)
    size = 1
    for dim in shape_tuple:
        size *= dim
    t: torch.Tensor = torch.ops.sgl_kernel.allocate_pin_memory(
        size, dtype, write_combined, numa_affinity
    )
    return t.view(shape_tuple)
