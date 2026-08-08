# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/xpu_communicator.py

from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_xpu


class XpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_xpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank_in_group = dist.get_rank(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def all_gatherv(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Variable-length all-gather on XCCL. Equal split uses the native
        collective; uneven emulates allgatherv with per-rank broadcasts (XCCL
        exposes no native allgatherv). Writes directly into `output`.
        """
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        if sizes is None:
            dist.all_gather_into_tensor(output, input_, group=self.group)
            return output

        assert len(sizes) == self.world_size
        # Per-rank blocking broadcasts. Each rank broadcasts its slice in turn.
        offset = 0
        for r, sz in enumerate(sizes):
            dst_slice = output[offset : offset + sz]
            # Seed my slice so the in-place broadcast sends real data.
            if r == self.rank_in_group:
                dst_slice.copy_(input_)
            dist.broadcast(
                dst_slice,
                src=dist.get_global_rank(self.group, r),
                group=self.group,
            )
            offset += sz
        return output

    def reduce_scatterv(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Variable-length reduce-scatter on XCCL (inverse of `all_gatherv`).
        Equal split uses the native collective; uneven runs a per-rank
        `dist.reduce` loop, targeting `output` on our own iteration and passing
        the input slice as a read-only send buffer on the others.
        """
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        if sizes is None:
            dist.reduce_scatter_tensor(output, input_, group=self.group)
            return output

        assert len(sizes) == self.world_size
        # Per-rank blocking reduce loop; grouped form deadlocks under
        # phase-divergent DP batches (see all_gatherv).
        offset = 0
        for r, sz in enumerate(sizes):
            global_r = dist.get_global_rank(self.group, r)
            if r == self.rank_in_group:
                # Own iteration: seed `output` with our chunk, then reduce
                # every other rank into it in place.
                output.copy_(input_[offset : offset + sz])
                dist.reduce(output, dst=global_r, group=self.group)
            else:
                # Contributor iteration: dist.reduce never writes to non-dst
                # ranks, so the input_ slice stays read-only.
                dist.reduce(
                    input_[offset : offset + sz],
                    dst=global_r,
                    group=self.group,
                )
            offset += sz
        return output

    def gather(
        self, input_: torch.Tensor, rank_in_group: int, dst: int = 0, dim: int = -1
    ):
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.group
        )
        if rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor
