from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclass
class _HostAllocation:
    indices: torch.Tensor
    num_tokens: int
    load_event: Any = None


class DecodeHostCache:
    """Own host destinations until the received KV has reached device memory."""

    def __init__(
        self,
        device_pool: MHATokenToKVPool | MLATokenToKVPool,
        page_size: int,
        host_pool: MHATokenToKVPoolHost | MLATokenToKVPoolHost,
        retraction_reserved_tokens: int,
        transfer_engine: L2TransferEngine,
    ):
        if type(device_pool) is MHATokenToKVPool:
            if (
                device_pool.kv_cache_layout != "nhd"
                or device_pool.v_head_dim != device_pool.head_dim
            ):
                raise ValueError("Decode host cache requires symmetric NHD MHA KV")
        elif type(device_pool) is not MLATokenToKVPool or device_pool.use_dsa:
            raise ValueError("Decode host cache requires dense MHA or plain MLA KV")
        if device_pool.layer_shard_enabled:
            raise ValueError("Decode host cache does not support layer-sharded KV")
        if host_pool.device_pool is not device_pool or host_pool.page_size != page_size:
            raise ValueError("Decode host cache must share the retraction KV pool")
        if host_pool.layout != "layer_first" or transfer_engine.io_backend != "kernel":
            raise ValueError(
                "Decode host cache requires layer_first layout and kernel IO"
            )
        if retraction_reserved_tokens < 0 or retraction_reserved_tokens % page_size:
            raise ValueError(
                "Retraction reservation must be a nonnegative page multiple"
            )
        if host_pool.logical_size < retraction_reserved_tokens + page_size:
            raise ValueError(
                "Decode host pool must hold a retraction reservation and one receive "
                "page; increase --hicache-size or --hicache-ratio"
            )

        self.device_pool = device_pool
        self.page_size = page_size
        self.host_pool = host_pool
        self.retraction_reserved_tokens = retraction_reserved_tokens
        self.transfer_engine = transfer_engine
        self.allocations: dict[Req, _HostAllocation] = {}

    def get_contiguous_buf_infos(self):
        if isinstance(self.host_pool, MLATokenToKVPoolHost):
            return self.host_pool.get_contiguous_buf_infos()
        buffers = self.host_pool.host_kv_data_refs
        return (
            [buffer.data_ptr() for buffer in buffers],
            [buffer.nbytes for buffer in buffers],
            [self.host_pool.token_stride_size * self.page_size] * len(buffers),
        )

    def allocate(self, req: Req, num_tokens: int) -> torch.Tensor | None:
        if req in self.allocations:
            raise ValueError("Request already owns decode host cache slots")
        if num_tokens <= 0:
            raise ValueError("Decode host cache allocation must contain tokens")
        required_tokens = ceil_align(num_tokens, self.page_size)
        if (
            self.host_pool.available_size() - required_tokens
            < self.retraction_reserved_tokens
        ):
            return None
        indices = self.host_pool.alloc(required_tokens)
        if indices is not None:
            self.allocations[req] = _HostAllocation(indices, num_tokens)
        return indices

    def contains(self, req: Req) -> bool:
        return req in self.allocations

    def load(self, reqs: list[Req], req_to_token_pool: ReqToTokenPool):
        """Enqueue H2D after the caller's forward fence and gate its current stream."""
        allocations = []
        device_indices = []
        for req in reqs:
            allocation = self.allocations.get(req)
            if allocation is None or allocation.load_event is not None:
                continue
            assert req.kv.req_pool_idx >= 0
            allocations.append(allocation)
            device_indices.append(
                req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, : allocation.num_tokens
                ]
            )
        if not allocations:
            return None

        host_indices = torch.cat(
            [allocation.indices[: allocation.num_tokens] for allocation in allocations]
        ).to(self.device_pool.device, non_blocking=True)
        completion = self.transfer_engine.submit_host_to_device(
            [
                L2Transfer(
                    host_pool=self.host_pool,
                    device_pool=self.device_pool,
                    host_indices=host_indices,
                    device_indices=torch.cat(device_indices).to(dtype=torch.int64),
                )
            ],
            layer_num=self.device_pool.layer_num,
        )
        for allocation in allocations:
            allocation.load_event = completion.finish_event
        completion.finish_event.wait()
        return completion.finish_event

    def release(self, req: Req) -> None:
        allocation = self.allocations.get(req)
        if allocation is None:
            return
        if allocation.load_event is not None:
            # Quick finish/abort may return GPU slots to an RDMA writer immediately.
            allocation.load_event.synchronize()
        self.host_pool.free(allocation.indices)
        del self.allocations[req]

    def poll(self, gloo_group: ProcessGroup | None = None) -> None:
        loading = [
            (req, allocation)
            for req, allocation in self.allocations.items()
            if allocation.load_event is not None
        ]
        completed = 0
        for _, allocation in loading:
            if not allocation.load_event.query():
                break
            completed += 1
        if gloo_group is not None and torch.distributed.get_world_size(gloo_group) > 1:
            # Host capacity must stay identical across ranks before the next admission.
            count = torch.tensor(completed, dtype=torch.int32)
            torch.distributed.all_reduce(
                count, op=torch.distributed.ReduceOp.MIN, group=gloo_group
            )
            completed = int(count.item())
        for req, allocation in loading[:completed]:
            self.host_pool.free(allocation.indices)
            del self.allocations[req]

    def clear(self) -> None:
        for req in list(self.allocations):
            self.release(req)
