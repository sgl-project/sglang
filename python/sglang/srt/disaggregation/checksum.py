from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.schedule_batch import Req


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
    allocator = scheduler.token_to_kv_pool_allocator
    kv_indices = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, start_idx:end_idx
    ]
    # The ids the transfer addressed; a virtual-id pool's raw row points elsewhere.
    kv_indices = allocator.translate_kv_indices_for_transfer(kv_indices)
    return _to_page_indices_gpu(kv_indices, allocator.page_size)


def _index_tensor(indices, device: torch.device) -> torch.Tensor:
    """One state transfer payload (tensor, array, int list, or a list of
    arrays) as a flat int64 tensor on `device`."""
    if isinstance(indices, torch.Tensor):
        flat = indices.reshape(-1)
    elif (
        isinstance(indices, (list, tuple))
        and len(indices) > 0
        and not np.isscalar(indices[0])
    ):
        return torch.cat([_index_tensor(part, device) for part in indices])
    else:
        flat = torch.from_numpy(np.asarray(indices, dtype=np.int64).reshape(-1))
    return flat.to(device=device, dtype=torch.int64).contiguous()


class KvChecksumComputer:
    def __init__(
        self,
        device: torch.device,
        kv_data_ptrs: Sequence[int],
        kv_item_lens: Sequence[int],
        state_data_ptrs: Sequence[Sequence[int]] = (),
        state_item_lens: Sequence[Sequence[int]] = (),
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        assert len(state_data_ptrs) == len(state_item_lens)
        self._device = torch.device(device)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        # One (ptrs, item_lens) per state component, in kv_args.state_types order;
        # the transfer addresses every buffer of a component by that component's
        # own index list, so the digest must too.
        self._state_components = [
            ([int(ptr) for ptr in ptrs], [int(item_len) for item_len in lens])
            for ptrs, lens in zip(state_data_ptrs, state_item_lens)
        ]
        for ptrs, lens in self._state_components:
            assert len(ptrs) == len(lens)
        assert self._kv_data_ptrs or any(ptrs for ptrs, _ in self._state_components)

    def compute(
        self,
        kv_page_indices,
        state_indices: Sequence[Optional[object]] = (),
    ) -> int:
        """Digest the KV pages and, per state component, the transfer payload
        that side sent or received for it; None skips a component that ships
        nothing for this request."""
        assert len(state_indices) == len(self._state_components), (
            f"{len(state_indices)} state payloads for "
            f"{len(self._state_components)} state components"
        )
        all_ptrs = list(self._kv_data_ptrs)
        all_lens = list(self._kv_item_lens)
        all_indices: List[torch.Tensor] = [
            _index_tensor(kv_page_indices, self._device)
        ] * len(self._kv_data_ptrs)
        for (ptrs, lens), indices in zip(self._state_components, state_indices):
            if indices is None:
                continue
            all_ptrs += ptrs
            all_lens += lens
            all_indices += [_index_tensor(indices, self._device)] * len(ptrs)
        if not all_indices:
            # Adler-32 of no bytes, as the kernel returns for empty indices.
            return 1
        return adler32_strided_checksum(all_ptrs, all_lens, all_indices)
