from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Sequence

import torch

from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)

NestedInts = Sequence[int] | Sequence[Sequence[int]]

# Bump when the digest's meaning changes, so peers on different builds compare
# signatures rather than digests and skip instead of aborting.
_SIGNATURE_VERSION = 1


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
    # As the send path does: on a unified-memory pool `req_to_token` holds
    # VIRTUAL ids while the registered buffers take physical ones.
    kv_indices = scheduler.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
        kv_indices
    )
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
        # Not translate_loc_from_full_to_swa: it returns kernel-facing ids
        # (unified_hybrid_swa.py:301) where the transfer needs physical (:337).
        window_swa = (
            scheduler.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
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
        page_size: int = 1,
    ):
        assert len(kv_data_ptrs) == len(kv_item_lens)
        assert len(kv_data_ptrs) > 0
        self._device = torch.device(device)
        self._kv_data_ptrs = [int(ptr) for ptr in kv_data_ptrs]
        self._kv_item_lens = [int(item_len) for item_len in kv_item_lens]
        self._state_data_ptrs = _flatten_ints(state_data_ptrs)
        self._state_item_lens = _flatten_ints(state_item_lens)
        assert len(self._state_data_ptrs) == len(self._state_item_lens)
        self.signature = self._compute_signature(page_size)

    def _compute_signature(self, page_size: int) -> int:
        """Identify the KV layout this digest is taken over.

        Two sides can only compare digests if they cover the same bytes, and
        several supported topologies mean they do not. A prefill with a
        different TP width has different per-page item lengths; a prefill whose
        pool covers a layer subset -- pipeline parallelism, or layer sharding;
        see `kv_args.prefill_start_layer` -- owns fewer buffers. Shipping this
        beside the digest turns those into one logged skip rather than a
        mismatch on every request.
        """
        parts = [
            str(_SIGNATURE_VERSION),
            str(page_size),
            str(len(self._kv_data_ptrs)),
            ",".join(str(x) for x in self._kv_item_lens),
            str(len(self._state_data_ptrs)),
            ",".join(str(x) for x in self._state_item_lens),
        ]
        digest = hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest()
        # Positive in the int64 metadata slot, and never 0 -- 0 is the
        # "no digest was written" sentinel.
        return (int.from_bytes(digest, "big") & ((1 << 62) - 1)) | 1

    def compute(
        self,
        kv_page_indices_gpu: torch.Tensor,
        state_indices_gpu: Optional[torch.Tensor] = None,
    ) -> int:
        assert kv_page_indices_gpu.is_cuda and kv_page_indices_gpu.is_contiguous()
        all_ptrs = list(self._kv_data_ptrs)
        all_lens = list(self._kv_item_lens)
        all_indices: List[torch.Tensor] = [kv_page_indices_gpu] * len(
            self._kv_data_ptrs
        )
        if self._state_data_ptrs:
            assert state_indices_gpu is not None
            assert state_indices_gpu.is_cuda and state_indices_gpu.is_contiguous()
            all_ptrs += self._state_data_ptrs
            all_lens += self._state_item_lens
            all_indices += [state_indices_gpu] * len(self._state_data_ptrs)
        return adler32_strided_checksum(all_ptrs, all_lens, all_indices)


def corrupt_one_kv_row_for_test(scheduler, kv_page_indices_gpu: torch.Tensor) -> bool:
    """Clobber one landed KV row, the way a slot reused mid-write would.

    Test-only, behind SGLANG_TEST_DISAGG_KV_CORRUPT_PROB. Writes through the
    pool's own buffers so the digest sees exactly what a real fault would
    leave behind. Returns whether anything was corrupted.
    """
    if kv_page_indices_gpu.numel() == 0:
        return False
    allocator = scheduler.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    # The indices are page ids; the K/V buffers are indexed by token slot
    # ([size, head_num, head_dim]), which only coincide at page_size == 1.
    slot = int(kv_page_indices_gpu[0].item()) * allocator.page_size
    for name in ("k_buffer", "kv_buffer", "v_buffer"):
        buffers = getattr(pool, name, None)
        if not buffers:
            continue
        buf = buffers[0]
        if slot >= buf.shape[0] or not buf.is_contiguous():
            continue
        # XOR over a byte view rather than arithmetic on the KV dtype: it is
        # guaranteed to change the bytes and works for fp8 and friends.
        buf.view(torch.uint8).reshape(buf.shape[0], -1)[slot] ^= 0xFF
        return True
    return False
