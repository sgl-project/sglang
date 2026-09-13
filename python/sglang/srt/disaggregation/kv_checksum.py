# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""End-to-end KV integrity check for a PD transfer.

The bootstrap_room field in the metadata buffer proves the *metadata* a decode
request committed came from the prefill that owned it. It says nothing about
the KV itself, which travels as a separate RDMA write into slots the decode
side pre-allocated: a slot freed and handed to another request while a write is
still in flight, or a stale sender writing after an abort, silently replaces
some of those rows and the request decodes against another request's KV.

So the prefill digests the KV it hands off and the decode re-digests what
landed in its own slots. The digest is keyed by each element's *logical*
coordinate (buffer index, position within the transferred range, word offset
in the row), never by slot id, so the two sides agree despite holding the rows
at completely different slots -- see ``kernels/ops/memory/kv_checksum.py``.

Cost control. Digesting every byte of every layer would roughly double the KV
memory traffic of a handoff. At ``SAMPLED`` level each request instead digests
``SGLANG_DISAGGREGATION_KV_CHECKSUM_BUFFERS`` whole buffers (full rows, so the
reads stay coalesced) out of the pool's list, chosen evenly spaced with a
per-request rotation derived from the bootstrap room. The fault this exists
for rewrites a whole row -- every layer of it -- so any nonempty sample sees
it; the rotation is there so that a stream of requests also sweeps the whole
model, catching a hypothetical fault confined to one layer.

Configuration agreement. Both sides derive a *layout signature* from the facts
that have to match for the two digests to be comparable at all (buffer count,
row width, sampling policy, version). The prefill ships its signature next to
the digest; a decode whose signature differs -- a different TP width, a PP
prefill, a different sampling setting, the feature off on one side -- skips the
comparison and says so once, instead of aborting every request.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import torch

from sglang.kernels.ops.memory.kv_checksum import (
    DEFAULT_TILE,
    META_BUFFERS,
    META_NUM_ROWS,
    META_SLOT_BASE,
    kv_slot_checksum,
)
from sglang.srt.environ import DisaggKVChecksumLevel, envs

logger = logging.getLogger(__name__)

# Bump whenever the digest definition changes, so a rolling upgrade sees a
# signature mismatch (check skipped) rather than a false corruption report.
KV_CHECKSUM_VERSION = 1

U64_MASK = (1 << 64) - 1

# Slots of the metadata buffer's bootstrap_room row that carry the KV digest.
# Slot 0 stays the bootstrap room itself; see MetadataBuffers.
ROOM_SLOT_CHECKSUM_SIG = 1
ROOM_SLOT_CHECKSUM_DIGEST = 2
ROOM_SLOT_CHECKSUM_TOKENS = 3


def _splitmix64(value: int) -> int:
    x = value & U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return (x ^ (x >> 31)) & U64_MASK


def _pool_kv_buffers(pool) -> Optional[List[torch.Tensor]]:
    """Slot-indexed KV buffers of ``pool``, or None if it has no such layout."""
    # Imported here: memory_pool pulls in the whole mem_cache stack.
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

    if isinstance(pool, BaseSWAKVPool):
        # A hybrid pool transfers its full-attention half addressed by the
        # request's own kv indices, and its sliding-window half under separate
        # window indices. Only the former is digestible here.
        return _pool_kv_buffers(pool.full_kv_pool)
    if isinstance(pool, MLATokenToKVPool):
        buffers = list(pool.kv_buffer)
    elif isinstance(pool, MHATokenToKVPool):
        buffers = list(pool.k_buffer) + list(pool.v_buffer)
    else:
        return None

    if not buffers:
        return None
    first = buffers[0]
    if first.numel() == 0 or not first.is_cuda:
        return None
    # A uniform row width is what lets one launch cover every buffer.
    if any(
        b.shape != first.shape or b.dtype != first.dtype or not b.is_contiguous()
        for b in buffers
    ):
        return None
    return buffers


def _unsupported_reason(pool, buffers: List[torch.Tensor]) -> Optional[str]:
    """Why ``buffers`` cannot stand in for what the transfer writes, or None.

    A subclass may keep KV that ``_pool_kv_buffers`` knows nothing about, and a
    packed pool may not store one contiguous row per slot. Both would digest
    something other than what crosses the wire, so both are rejected here
    rather than risking a digest that disagrees on healthy traffic.
    """
    row_bytes = buffers[0][0].nbytes
    if row_bytes % 4 != 0:
        return f"KV row of {row_bytes} bytes is not 4-byte aligned"

    try:
        ptrs, _, item_lens = pool.get_contiguous_buf_infos()
    except NotImplementedError as e:
        return f"{type(pool).__name__} does not expose contiguous KV regions ({e})"

    page_size = pool.page_size
    if [b.data_ptr() for b in buffers] != list(ptrs):
        return (
            f"{type(pool).__name__} transfers KV regions other than the ones "
            f"this would digest"
        )
    if any(item_len != page_size * row_bytes for item_len in item_lens):
        return (
            f"{type(pool).__name__} does not store one contiguous row per slot "
            f"(page_size={page_size}, row_bytes={row_bytes}, "
            f"item_lens={sorted(set(item_lens))})"
        )
    return None


class KVChecksummer:
    """Digests the KV rows of PD requests, batched into one kernel launch."""

    def __init__(
        self,
        *,
        buffers: List[torch.Tensor],
        level: DisaggKVChecksumLevel,
        num_sampled_buffers: int,
    ):
        self.buffers = buffers
        self.level = level
        self.num_buffers = len(buffers)
        self.row_bytes = buffers[0][0].nbytes
        self.num_sampled_buffers = (
            self.num_buffers
            if level >= DisaggKVChecksumLevel.FULL
            else max(1, min(num_sampled_buffers, self.num_buffers))
        )
        self.device = buffers[0].device

        from sglang.kernels.ops.memory.ptr_table import make_ptr_table

        self.tile = DEFAULT_TILE
        self._ptr_table = make_ptr_table(
            [b.data_ptr() for b in buffers], device=self.device
        )
        # Evenly spaced picks; the per-request rotation is added on top.
        self._sample_offsets = (
            np.arange(self.num_sampled_buffers, dtype=np.int64) * self.num_buffers
        ) // self.num_sampled_buffers
        # Reused pinned staging for the metadata table (grown on demand), with
        # an event guarding its reuse -- see _meta_rows.
        self._meta_stride = META_BUFFERS + self.num_sampled_buffers
        self._staging: Optional[torch.Tensor] = None
        self._staging_np: Optional[np.ndarray] = None
        self._staging_event = torch.cuda.Event()
        self._staging_event.record()
        # Everything a comparison depends on. Two sides that agree here hold
        # byte-comparable rows; two that do not must not compare digests.
        signature = 0
        for field in (
            KV_CHECKSUM_VERSION,
            self.num_buffers,
            self.row_bytes,
            self.num_sampled_buffers,
            self.tile,
        ):
            signature = _splitmix64(signature ^ _splitmix64(field))
        # Slot 0 of the digest triple means "no digest", so never hand back 0.
        self.signature = signature | 1

    @classmethod
    def maybe_create(
        cls, token_to_kv_pool, *, enabled: bool = True
    ) -> Optional[KVChecksummer]:
        """Build a checksummer for this pool, or None if off or unsupported.

        Returning None is always safe: this side simply ships no digest, and
        the other side has nothing to compare against and says so once.
        """
        from sglang.srt.runtime_context import get_parallel

        level = DisaggKVChecksumLevel(envs.SGLANG_DISAGGREGATION_KV_CHECKSUM.get())
        if level is DisaggKVChecksumLevel.OFF or not enabled:
            return None

        parallel = get_parallel()
        if parallel.dcp_size > 1 or parallel.attn_cp_size > 1:
            # Context parallelism spreads one sequence's KV across ranks, so
            # which rows a rank holds is not simply its own kv indices.
            logger.warning(
                "PD KV checksum disabled: context parallelism "
                "(dcp_size=%d, attn_cp_size=%d) is not supported.",
                parallel.dcp_size,
                parallel.attn_cp_size,
            )
            return None

        buffers = _pool_kv_buffers(token_to_kv_pool)
        if buffers is None:
            logger.warning(
                "PD KV checksum disabled: %s has no slot-indexed KV buffers.",
                type(token_to_kv_pool).__name__,
            )
            return None
        reason = _unsupported_reason(token_to_kv_pool, buffers)
        if reason is not None:
            logger.warning("PD KV checksum disabled: %s.", reason)
            return None

        self = cls(
            buffers=buffers,
            level=level,
            num_sampled_buffers=envs.SGLANG_DISAGGREGATION_KV_CHECKSUM_BUFFERS.get(),
        )
        logger.info(
            "PD KV checksum enabled: level=%s buffers=%d sampled_per_req=%d "
            "row_bytes=%d signature=0x%016x",
            level.name,
            self.num_buffers,
            self.num_sampled_buffers,
            self.row_bytes,
            self.signature,
        )
        return self

    def _sampled_buffer_ids_batch(self, rooms: Sequence[int]) -> np.ndarray:
        """The buffers each request digests, ``[len(rooms), num_sampled]``.

        Evenly spaced across the pool's buffer list, rotated by the request's
        bootstrap room: one request pays for a small sample, while a stream of
        them sweeps every buffer. Both sides of a transfer know the room, so
        they pick the same set without negotiating.
        """
        n = self.num_buffers
        bases = np.fromiter(
            (_splitmix64(room) % n for room in rooms), dtype=np.int64, count=len(rooms)
        )
        return (bases[:, None] + self._sample_offsets) % n

    def sampled_buffer_ids(self, room: int) -> np.ndarray:
        """The buffers one request digests. Exposed for tests and diagnostics."""
        return self._sampled_buffer_ids_batch([room])[0].astype(np.int32)

    def _meta_rows(self, num_reqs: int) -> np.ndarray:
        """Writable host rows of the metadata table, in reused pinned memory.

        One row per request and contiguous, so the copy to device is a straight
        DMA rather than a compacting one. Waits out the previous call's copy
        first: from pinned memory that copy is asynchronous, so overwriting the
        rows while it is still in flight would corrupt it. In practice the
        caller has already read the previous digest, and the wait is a no-op.
        """
        if self._staging is None or self._staging.shape[0] < num_reqs:
            self._staging = torch.empty(
                (max(num_reqs, 64), self._meta_stride),
                dtype=torch.int32,
                pin_memory=True,
            )
            self._staging_np = self._staging.numpy()
        else:
            self._staging_event.synchronize()
        return self._staging_np[:num_reqs]

    def compute(
        self,
        rooms: Sequence[int],
        slot_tensors: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Digest each request's rows. Returns int64 ``[len(rooms)]``, unsynced.

        The per-request metadata row is all the host builds: the kernel derives
        every row's position and slot from it, so the cost here does not grow
        with the number of tokens digested.
        """
        assert len(rooms) == len(slot_tensors)
        num_reqs = len(rooms)
        out = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        lengths = np.fromiter(
            (t.numel() for t in slot_tensors), dtype=np.int32, count=num_reqs
        )
        max_rows = int(lengths.max()) if num_reqs else 0
        if max_rows == 0:
            return out

        meta = self._meta_rows(num_reqs)
        meta[:, META_NUM_ROWS] = lengths
        np.cumsum(lengths[:-1], out=meta[1:, META_SLOT_BASE])
        meta[0, META_SLOT_BASE] = 0
        meta[:, META_BUFFERS:] = self._sampled_buffer_ids_batch(rooms)

        device_meta = self._staging[:num_reqs].to(self.device, non_blocking=True)
        self._staging_event.record()

        nonempty = [t for t in slot_tensors if t.numel() > 0]
        slots = nonempty[0] if len(nonempty) == 1 else torch.cat(nonempty)
        return kv_slot_checksum(
            buf_ptr_table=self._ptr_table,
            row_bytes=self.row_bytes,
            meta=device_meta,
            slots=slots,
            num_sampled=self.num_sampled_buffers,
            max_rows=max_rows,
            tile=self.tile,
            out=out,
        )

    def compute_one(self, room: int, slots: torch.Tensor) -> torch.Tensor:
        return self.compute([room], [slots])

    def corrupt_rows_for_test(self, slots: torch.Tensor, num_rows: int = 1) -> None:
        """Clobber whole KV rows the way a reused slot would, for tests.

        Every buffer of the chosen rows is overwritten, which is what a stray
        write into a reused slot does -- and what makes a sampled digest a
        sound detector for it.
        """
        if slots.numel() == 0:
            return
        picks = torch.randint(
            0, slots.numel(), (min(num_rows, slots.numel()),), device=slots.device
        )
        rows = slots[picks].to(torch.int64)
        for buffer in self.buffers:
            buffer[rows] = torch.randn_like(buffer[rows].to(torch.float32)).to(
                buffer.dtype
            )


def digest_to_u64(value: int) -> int:
    """The kernel accumulates in int64; the metadata buffer stores uint64."""
    return value & U64_MASK


def u64_to_i64(value: int) -> int:
    """Same bits, signed. torch rejects a >= 2**63 Python int even for a uint64
    tensor ("Overflow when unpacking long long"), so writes go through an
    int64 view of the buffer."""
    value &= U64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value
