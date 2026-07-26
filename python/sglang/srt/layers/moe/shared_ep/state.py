"""Owning SharedEP VMM state and typed tensor views."""

from __future__ import annotations

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.layers.moe.shared_ep.epoch import GpuEpoch, create_gpu_epoch
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpInputViews,
    SharedEpLayout,
    align_output_layout,
)
from sglang.srt.layers.moe.shared_ep.vmm import (
    SharedEpVmmAllocation,
    allocate_rank_major_vmm,
)


class SharedEpState(msgspec.Struct, kw_only=True):
    """Process-lifetime mappings; ``close`` supports partial setup cleanup."""

    layout: SharedEpLayout
    input_allocation: SharedEpVmmAllocation
    output_allocation: SharedEpVmmAllocation
    input_epoch: GpuEpoch
    output_epoch: GpuEpoch
    global_input: SharedEpInputViews | None
    local_input: SharedEpInputViews | None
    global_output: torch.Tensor | None
    local_output: torch.Tensor | None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.global_input = None
        self.local_input = None
        self.global_output = None
        self.local_output = None
        self.output_epoch.close()
        self.input_epoch.close()
        self.output_allocation.close()
        self.input_allocation.close()


def create_shared_ep_state(
    *,
    layout: SharedEpLayout,
    cpu_group,
    device: torch.device,
) -> SharedEpState:
    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    resources = []
    try:
        input_allocation = allocate_rank_major_vmm(
            cpu_group=cpu_group,
            device=device,
            logical_rank_bytes=layout.input_rank_bytes,
        )
        resources.append(input_allocation)
        layout = align_output_layout(
            layout,
            granularity=input_allocation.granularity,
        )
        output_allocation = allocate_rank_major_vmm(
            cpu_group=cpu_group,
            device=device,
            logical_rank_bytes=layout.output_rank_bytes,
        )
        resources.append(output_allocation)
        if output_allocation.mapped_rank_bytes != layout.output_rank_bytes:
            raise RuntimeError(
                "SharedEP output rank stride must be exactly representable by rows"
            )
        input_epoch = create_gpu_epoch(
            cpu_group=cpu_group,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        resources.append(input_epoch)
        output_epoch = create_gpu_epoch(
            cpu_group=cpu_group,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        resources.append(output_epoch)

        global_input = layout.input_views(
            input_allocation.global_storage,
            world_size=world_size,
            mapped_rank_bytes=input_allocation.mapped_rank_bytes,
        )
        local_input = layout.input_views(
            input_allocation.local_storage,
            world_size=1,
            mapped_rank_bytes=input_allocation.mapped_rank_bytes,
        ).owner(0)
        global_output = layout.output_view(
            output_allocation.global_storage,
            world_size=world_size,
            mapped_rank_bytes=output_allocation.mapped_rank_bytes,
        )
        local_output = layout.output_view(
            output_allocation.local_storage,
            world_size=1,
            mapped_rank_bytes=output_allocation.mapped_rank_bytes,
        )[0]
        return SharedEpState(
            layout=layout,
            input_allocation=input_allocation,
            output_allocation=output_allocation,
            input_epoch=input_epoch,
            output_epoch=output_epoch,
            global_input=global_input,
            local_input=local_input,
            global_output=global_output,
            local_output=local_output,
        )
    except BaseException:
        while resources:
            resources.pop().close()
        raise
