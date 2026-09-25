from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

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


def page_indices_for_request(scheduler, req: Req, end_idx: int) -> torch.Tensor:
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    kv_indices = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, 0:end_idx
    ]
    return _to_page_indices_gpu(kv_indices, page_size)


def state_indices_for_request(
    scheduler, req: Req, seq_len: int
) -> Optional[torch.Tensor]:
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if isinstance(pool, HybridLinearKVPool):
        return (
            scheduler.req_to_token_pool.req_index_to_mamba_index_mapping[
                req.kv.req_pool_idx
            ]
            .to(torch.int64)
            .contiguous()
            .reshape(-1)
        )
    if isinstance(pool, SWAKVPool):
        page_size = scheduler.token_to_kv_pool_allocator.page_size
        window_size = scheduler.sliding_window_size
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        window_full = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, window_start:seq_len
        ]
        window_swa = (
            scheduler.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_full
            )
        )
        return _to_page_indices_gpu(window_swa, page_size)
    if isinstance(pool, DSATokenToKVPool):
        device_page_size = pool.page_size
        kv_full = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :seq_len
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
        state_types: Optional[Sequence[str]] = None,
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        assert len(kv_data_ptrs) > 0
        self._device = torch.device(device)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        self._state_data_ptrs = _flatten_ints(state_data_ptrs)
        self._state_item_lens = _flatten_ints(state_item_lens)
        assert len(self._state_data_ptrs) == len(self._state_item_lens)
        if self._device.type == "npu" and state_types:
            # compute() currently supplies one index vector for all state
            # descriptors. Heterogeneous components (e.g. DSV4 C128 and C4
            # ring state) require separate index mappings before flattening.
            if set(state_types) not in ({"mamba"}, {"swa"}):
                raise NotImplementedError(
                    "NPU KV checksum requires a shared Mamba or SWA state index "
                    f"mapping; unsupported state components: {list(state_types)}"
                )

    def compute(
        self,
        kv_page_indices_gpu: torch.Tensor,
        state_indices_gpu: Optional[torch.Tensor] = None,
    ) -> int:
        expected_device = self._device.type
        assert expected_device in ("cuda", "npu")
        assert kv_page_indices_gpu.device.type == expected_device
        if self._device.index is not None:
            assert kv_page_indices_gpu.device == self._device
        assert kv_page_indices_gpu.is_contiguous()
        all_ptrs = list(self._kv_data_ptrs)
        all_lens = list(self._kv_item_lens)
        all_indices: List[torch.Tensor] = [kv_page_indices_gpu] * len(
            self._kv_data_ptrs
        )
        if self._state_data_ptrs:
            assert state_indices_gpu is not None
            assert state_indices_gpu.device == kv_page_indices_gpu.device
            assert state_indices_gpu.is_contiguous()
            all_ptrs += self._state_data_ptrs
            all_lens += self._state_item_lens
            all_indices += [state_indices_gpu] * len(self._state_data_ptrs)
        if expected_device == "npu":
            from sglang.srt.hardware_backend.npu.checksum import (
                adler32_strided_checksum,
            )
        else:
            from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum

        return adler32_strided_checksum(all_ptrs, all_lens, all_indices)
