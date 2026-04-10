from typing import List

import torch
import torch.distributed as dist


def calc_patch_height_index(patch_height_list: List[torch.Tensor]) -> torch.Tensor:
    patch_heights = torch.stack(patch_height_list).view(-1)
    return torch.cat([patch_heights.new_zeros(1), torch.cumsum(patch_heights, dim=0)])


def calc_top_halo_size(
    local_rank, world_size, patch_height_index, kernel_size, padding, stride
):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if local_rank == 0:
        return 0
    nstep_before_top = (
        patch_height_index[local_rank] + padding - (kernel_size - 1) // 2 + stride - 1
    ) // stride
    top_halo_size = patch_height_index[local_rank] - (
        nstep_before_top * stride - padding
    )
    return top_halo_size


def calc_bottom_halo_size(
    local_rank, world_size, patch_height_index, kernel_size, padding, stride
):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if local_rank == world_size - 1:
        return 0
    nstep_before_bottom = (
        patch_height_index[local_rank + 1]
        + padding
        - (kernel_size - 1) // 2
        + stride
        - 1
    ) // stride
    bottom_halo_size = (
        (nstep_before_bottom - 1) * stride
        + kernel_size
        - padding
        - patch_height_index[local_rank + 1]
    )
    return bottom_halo_size


def halo_exchange(
    x: torch.Tensor,
    *,
    rank: int,
    group,
    prev_bottom_halo_size: int,
    next_top_halo_size: int,
    curr_top_halo_size: int,
    curr_bottom_halo_size: int,
) -> torch.Tensor:
    b, c, t, _, w = x.shape
    device = x.device

    comm_ops = []
    top_halo_recv = None
    bottom_halo_recv = None

    if prev_bottom_halo_size > 0:
        top_halo_send = x[:, :, :, :prev_bottom_halo_size, :].contiguous()
        comm_ops.append(dist.P2POp(dist.isend, top_halo_send, rank - 1, group=group))

    if next_top_halo_size > 0:
        bottom_halo_send = x[:, :, :, -next_top_halo_size:, :].contiguous()
        comm_ops.append(dist.P2POp(dist.isend, bottom_halo_send, rank + 1, group=group))

    if curr_top_halo_size > 0:
        top_halo_recv = torch.empty(
            [b, c, t, curr_top_halo_size, w], dtype=x.dtype, device=device
        )
        comm_ops.append(dist.P2POp(dist.irecv, top_halo_recv, rank - 1, group=group))
    elif curr_top_halo_size < 0:
        x = x[:, :, :, -curr_top_halo_size:, :]

    if curr_bottom_halo_size > 0:
        bottom_halo_recv = torch.empty(
            [b, c, t, curr_bottom_halo_size, w], dtype=x.dtype, device=device
        )
        comm_ops.append(dist.P2POp(dist.irecv, bottom_halo_recv, rank + 1, group=group))
    elif curr_bottom_halo_size < 0:
        x = x[:, :, :, :curr_bottom_halo_size, :]

    if comm_ops:
        reqs = dist.batch_isend_irecv(comm_ops)
        for req in reqs:
            req.wait()

    if top_halo_recv is not None:
        x = torch.cat([top_halo_recv, x], dim=-2)
    if bottom_halo_recv is not None:
        x = torch.cat([x, bottom_halo_recv], dim=-2)

    return x
