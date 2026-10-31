"""Layer-owned quantized latent shards for the merged Kimi front projection."""

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.all_reduce import Communicator, PullPlane
from sglang.kernels.ops.moe.moe_route_quant_fused import (
    QuantGather,
    available,
    route_quant_fused,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    _SEMAPHORE_BYTES,
    _allocate_symmetric_memory,
)
from sglang.srt.runtime_context import get_buffer

# Serving wins measured for single-token decode and an eight-token verify block.
# The gather kernel itself takes a runtime row count.
FRONT_SHARD_ROWS = (1, 8)


def sharded_front_available(group, hidden_size: int) -> bool:
    def check():
        # All TP ranks must select the same collective path.
        ready = torch.tensor(
            [available(group.world_size, hidden_size)], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
        return bool(ready.item())

    return get_buffer(
        f"k3_front_gather_available_{group.unique_name}_{hidden_size}", check
    )


class KimiK3FrontWorkspace:
    def __init__(self, group, num_layers: int, hidden_size: int, max_tokens: int):
        device = torch.device("cuda", torch.cuda.current_device())
        self.rank = group.rank_in_group
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        local_groups = hidden_size // group.world_size // 32
        scale_stride = (local_groups + 15) // 16 * 16
        self.scale_row_bytes = group.world_size * scale_stride
        scale_start = num_layers * max_tokens * hidden_size
        scale_end = scale_start + num_layers * max_tokens * self.scale_row_bytes
        sem_start = (
            (scale_end + _SEMAPHORE_BYTES - 1) // _SEMAPHORE_BYTES * _SEMAPHORE_BYTES
        )
        num_blocks = 2 * max_tokens
        total = sem_start + num_blocks * _SEMAPHORE_BYTES
        with torch.inference_mode(False):
            self.region, self.handle = _allocate_symmetric_memory(
                total, device, group.cpu_group
            )
        self.region[sem_start:].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)
        peers = [
            self.handle.get_buffer(i, [total], torch.uint8)[sem_start:].view(
                num_blocks, _SEMAPHORE_BYTES
            )
            for i in range(group.world_size)
        ]
        mc_base = int(self.handle.multicast_ptr)
        # Separate from the shared expert's overlapping AllReduce semaphore plane.
        self.comm = Communicator(
            pull=PullPlane(
                self.rank,
                group.world_size,
                semaphores=peers,
                mc_semaphore=mc_base + sem_start,
            )
        )
        self.quantized = (
            self.region[:scale_start]
            .view(torch.float8_e4m3fn)
            .view(num_layers, max_tokens, hidden_size)
        )
        self.scales = (
            self.region[scale_start:scale_end]
            .view(torch.int32)
            .view(num_layers, max_tokens, group.world_size, scale_stride // 4)
        )
        self.q_mc = mc_base
        self.scale_mc = mc_base + scale_start

    def route(self, layer_id, scores, bias, local_x, config):
        rows = scores.shape[0]
        assert rows <= self.max_tokens
        # Finalize/AllReduce fences the consumer before this layer reuses its slot.
        gather = QuantGather(
            self.comm,
            self.quantized[layer_id, :rows],
            self.scales[layer_id, :rows],
            self.q_mc + layer_id * self.max_tokens * self.hidden_size,
            self.scale_mc + layer_id * self.max_tokens * self.scale_row_bytes,
        )
        return route_quant_fused(
            scores,
            bias,
            local_x,
            config.top_k,
            routed_scaling_factor=(
                config.routed_scaling_factor
                if config.routed_scaling_factor is not None
                else 1.0
            ),
            renormalize=config.renormalize,
            apply_scale=config.apply_routed_scaling_factor_on_output,
            gather=gather,
        )
