from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MiniMaxSparseKVPool

NestedInts = Sequence[int] | Sequence[Sequence[int]]


def _flatten_ints(values: NestedInts) -> list[int]:
    flattened: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(int(item) for item in value)
        else:
            flattened.append(int(value))
    return flattened


def is_health_check_req(req: Req) -> bool:
    rid = req.rid
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def _to_page_indices_gpu(idx: torch.Tensor, page_size: int) -> torch.Tensor:
    idx = idx.to(torch.int64).contiguous().reshape(-1)
    if page_size == 1:
        return idx
    return (idx[::page_size] // page_size).contiguous()


def page_indices_for_request(
    scheduler, req: Req, end_idx: int, start_idx: int = 0
) -> torch.Tensor:
    # start_idx is the page-aligned prefix decode already held, which prefill
    # never sent; both sides digest [start_idx, end_idx) of their own row.
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    kv_indices = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, start_idx:end_idx
    ]
    return _to_page_indices_gpu(kv_indices, page_size)


def _window_rows(scheduler, req: Req, ids, window, seq_len, start_idx):
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    window_start = max(start_idx, seq_len - window)
    window_start = (window_start // page_size) * page_size
    window_full = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, window_start:seq_len
    ]
    return _to_page_indices_gpu(ids.ids(window_full), page_size)


def state_indices_for_request(
    scheduler, req: Req, seq_len: int, start_idx: int = 0
) -> Union[None, torch.Tensor, List[torch.Tensor]]:
    """The state rows the transfer of ``req`` digests: one tensor for every
    entry, or one per entry for a unified pool of several windows."""
    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

    storage = kv_storage_of(
        scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
    )
    sub_pools = storage.entry_sub_pools()
    # Only a storage whose entries are sub-pools keeps a tail per window.
    windows = storage.retained_window_lengths() if sub_pools else {}
    if any(name in windows for name in sub_pools):
        # Each entry digests the units the transfer sends of its own sub-pool:
        # a window its kept tail, a recurrent state the request's slots.
        state = set(storage.state_sub_pools())
        return [
            (
                storage.state_unit_ids(name, req)
                if name in state
                else storage.resume_units(
                    name, req, token_start=start_idx, token_boundary=seq_len
                ).unit_ids
            )
            for name in sub_pools
        ]
    return _state_indices(scheduler, req, seq_len, start_idx)


def _state_indices(
    scheduler, req: Req, seq_len: int, start_idx: int = 0
) -> Optional[torch.Tensor]:
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    router = pool.layer_router
    # A pool wrapping one token KV view keeps its other layers' state per request.
    if router.wrapped_view() is not None:
        return (
            scheduler.req_to_token_pool.req_index_to_mamba_index_mapping[
                req.kv.req_pool_idx
            ]
            .to(torch.int64)
            .contiguous()
            .reshape(-1)
        )
    if router.sub_pool(IdSpaceKind.SLIDING_WINDOW) is not None:
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

        tails = kv_storage_of(
            scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
        ).retained_window_lengths()
        return _window_rows(
            scheduler,
            req,
            scheduler.token_to_kv_pool_allocator.id_spaces()[
                router.sub_pool(IdSpaceKind.SLIDING_WINDOW).name
            ],
            max(tails.values()),
            seq_len,
            start_idx,
        )
    if isinstance(pool, (DSATokenToKVPool, MiniMaxSparseKVPool)):
        device_page_size = pool.page_size
        kv_full = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, start_idx:seq_len
        ]
        return _to_page_indices_gpu(kv_full, device_page_size)
    return None


class KvChecksumComputer:
    def __init__(
        self,
        device: torch.device,
        kv_data_ptrs: Sequence[int],
        kv_item_lens: Sequence[int],
        state_data_ptrs: NestedInts = (),
        state_item_lens: NestedInts = (),
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        self._device = torch.device(device)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        self._state_data_ptrs = _flatten_ints(state_data_ptrs)
        self._state_item_lens = _flatten_ints(state_item_lens)
        # How many of the flattened state buffers each state entry holds.
        self._state_entry_sizes = [
            len(value) if isinstance(value, (list, tuple)) else 1
            for value in state_data_ptrs
        ]
        assert len(self._state_data_ptrs) == len(self._state_item_lens)
        assert self._kv_data_ptrs or self._state_data_ptrs

    def compute(
        self,
        kv_page_indices_gpu: torch.Tensor,
        state_indices_gpu: Union[None, torch.Tensor, Sequence[torch.Tensor]] = None,
    ) -> int:
        """``state_indices_gpu`` is one tensor for every state buffer, or one
        per state entry (`state_indices_for_request`)."""
        assert kv_page_indices_gpu.is_cuda and kv_page_indices_gpu.is_contiguous()
        all_ptrs = list(self._kv_data_ptrs)
        all_lens = list(self._kv_item_lens)
        all_indices: List[torch.Tensor] = [kv_page_indices_gpu] * len(
            self._kv_data_ptrs
        )
        if self._state_data_ptrs:
            assert state_indices_gpu is not None
            if isinstance(state_indices_gpu, torch.Tensor):
                per_entry = [state_indices_gpu] * len(self._state_entry_sizes)
            else:
                per_entry = list(state_indices_gpu)
                assert len(per_entry) == len(self._state_entry_sizes)
            for indices, size in zip(per_entry, self._state_entry_sizes):
                assert indices.is_cuda and indices.is_contiguous()
                all_indices += [indices] * size
            all_ptrs += self._state_data_ptrs
            all_lens += self._state_item_lens
        return adler32_strided_checksum(all_ptrs, all_lens, all_indices)
