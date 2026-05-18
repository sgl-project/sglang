from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.kv_cache_canary.fingerprint import splitmix64_mix, to_signed_int64

if TYPE_CHECKING:
    from tvm_ffi.module import Module


VIOLATION_FIELDS: int = 8
CANARY_FIELDS_PER_SLOT: int = 5
CANARY_SLOT_BYTES: int = CANARY_FIELDS_PER_SLOT * 8

# Slot field offsets (in 8-byte words). Mirrors the kCanaryField* constants
# in ``canary.cuh``.
_CANARY_FIELD_REQ_ID: int = 0
_CANARY_FIELD_TOKEN_ID: int = 1
_CANARY_FIELD_POSITION: int = 2
_CANARY_FIELD_PREV_HASH: int = 3
# real_kv_hash: splitmix64 mix of a few bytes of the real KV pool's slot
# data captured at write time. Verified by reading the same bytes again
# at verify time; mismatch implies the real KV slot changed underneath
# the canary (the attn-kernel-config / PD-transfer corruption modes
# canary-with-real-data is designed to catch).
_CANARY_FIELD_REAL_KV_HASH: int = 4

# Modes for ``--kv-cache-canary-real-data``. ``OFF`` disables the
# real-KV mix entirely (the real_kv_hash field stays zero); ``BIT`` mixes
# a 16-byte prefix of the real slot; ``ALL`` mixes the full real-slot
# stride. Mirrored in C++ as ``kRealKvHashMode*`` constants.
REAL_KV_HASH_MODE_OFF: int = 0
REAL_KV_HASH_MODE_BIT: int = 1
REAL_KV_HASH_MODE_ALL: int = 2
REAL_KV_HASH_BIT_BYTES: int = 16

# Violation-row field offsets. Mirrors the kViolationField* constants.
_VIOLATION_FIELD_KERNEL_KIND: int = 0
_VIOLATION_FIELD_FAIL_REASON: int = 1
_VIOLATION_FIELD_SLOT_IDX: int = 2
_VIOLATION_FIELD_REQ_ID: int = 3
_VIOLATION_FIELD_TOKEN_ID: int = 4
_VIOLATION_FIELD_POSITION: int = 5
_VIOLATION_FIELD_EXPECTED_HASH: int = 6
_VIOLATION_FIELD_ACTUAL_HASH: int = 7

KERNEL_KIND_HEAD: int = 0
KERNEL_KIND_TAIL: int = 1

_U64_MASK: int = (1 << 64) - 1


class FailReason(enum.IntEnum):
    """Mirror of C++ ``kFailReason*`` constants in ``canary.cuh``.

    Single source of truth — kernel and Python must agree on these integers.
    """

    NONE = 0
    REQ_ID = 1
    TOKEN_ID = 2
    POSITION = 3
    HASH = 4
    POSITION_MONOTONIC = 5
    REAL_KV_HASH = 6


@cache_once
def _jit_canary_module() -> "Module":
    return load_jit(
        "kv_cache_canary",
        cuda_files=["kv_cache_canary/canary.cuh"],
        cuda_wrappers=[("canary_step", "canary_step")],
    )


def canary_step(
    *,
    src_buf: torch.Tensor,
    dst_buf: torch.Tensor,
    slot_stride_bytes: int,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_req_ids: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    verify_active_mask: torch.Tensor,
    write_slot_indices: torch.Tensor,
    write_token_ids: torch.Tensor,
    write_positions: torch.Tensor,
    write_req_ids: torch.Tensor,
    write_req_seed_slot_indices: torch.Tensor,
    write_req_entry_starts: torch.Tensor,
    write_req_entry_counts: torch.Tensor,
    write_req_active_mask: torch.Tensor,
    seed: int,
    violation_ring: torch.Tensor,
    violation_ring_valid: torch.Tensor,
    violation_write_index: torch.Tensor,
    first_violation: torch.Tensor,
    first_violation_set: torch.Tensor,
    is_errored: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    kernel_kind: int,
    real_kv_buf: torch.Tensor,
    real_kv_slot_stride_bytes: int,
    real_kv_read_bytes: int,
    real_kv_hash_mode: int,
) -> None:
    """Launch one KV-cache canary step.

    The kernel runs two distinct workloads sharing one grid:

    1. **Verify entries** (per-thread): read a slot's stored fingerprint,
       compute the expected ``prev_hash`` by reading the previous slot and
       running ``splitmix64_mix`` on-device, then compare against three
       expected fields (``req_id``, ``position``, ``prev_hash``) — one
       independent fail_reason per field. ``token_id`` is NOT verified on
       this path (historical tokens are not in the current
       ``ForwardBatch.input_ids``); token tampering is caught indirectly
       via the splitmix64 chain at the next position.
    2. **Write-req chains** (per-thread, one driver thread per request): walk
       the splitmix64 chain across all writes for that req, seeded from
       either ``kSeed`` (``write_req_seed_slot_indices == -1``) or the
       canary slot at ``K_req - 1``, and store ``(req_id, token_id, position,
       prev_hash)`` into each slot in order.

    The host emits **raw input data** (``token_id``, ``position``) from the
    sglang ``ForwardBatch`` — no host-side hash computation. ``seed`` is
    ``CanaryConfig.seed``.

    Args:
        src_buf:             uint8 flatten view of the source shadow tensor
                             (verify reads + prev-slot reads here).
        dst_buf:             uint8 flatten view of the destination shadow.
        slot_stride_bytes:   bytes per logical slot in both buffers.
        verify_*:            per-verify-entry arrays. ``verify_prev_slot_indices``
                             at ``-1`` means "position 0, expected prev_hash =
                             kSeed". ``verify_active_mask`` at 0 means
                             "skip-sentinel padding for cuda graph".
        write_*:             per-write-entry arrays (pure data; driver threads
                             read these).
        write_req_*:         per-write-req arrays, one row per req with new
                             writes. The driver thread walks
                             ``[entry_starts, entry_starts + entry_counts)``
                             sequentially, advancing splitmix64.
        seed:                ``CanaryConfig.seed`` (chain head).
        violation_*:         GPU ring + first-violation latch + is_errored
                             flag for error reporting.
        slot_run_counter:    int64 [1]. ``+= num_active_slots`` per launch.
        kernel_run_counter:  int64 [1]. ``+= 1`` per launch.
        kernel_kind:         ``KERNEL_KIND_HEAD`` or ``KERNEL_KIND_TAIL``.
    """
    module = _jit_canary_module()
    module.canary_step(
        src_buf,
        dst_buf,
        slot_stride_bytes,
        verify_slot_indices,
        verify_positions,
        verify_req_ids,
        verify_prev_slot_indices,
        verify_active_mask,
        write_slot_indices,
        write_token_ids,
        write_positions,
        write_req_ids,
        write_req_seed_slot_indices,
        write_req_entry_starts,
        write_req_entry_counts,
        write_req_active_mask,
        to_signed_int64(int(seed)),
        violation_ring,
        violation_ring_valid,
        violation_write_index,
        first_violation,
        first_violation_set,
        is_errored,
        slot_run_counter,
        kernel_run_counter,
        kernel_kind,
        real_kv_buf,
        real_kv_slot_stride_bytes,
        real_kv_read_bytes,
        real_kv_hash_mode,
    )


def canary_step_torch_reference(
    *,
    src_buf: torch.Tensor,
    dst_buf: torch.Tensor,
    slot_stride_bytes: int,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_req_ids: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    verify_active_mask: torch.Tensor,
    write_slot_indices: torch.Tensor,
    write_token_ids: torch.Tensor,
    write_positions: torch.Tensor,
    write_req_ids: torch.Tensor,
    write_req_seed_slot_indices: torch.Tensor,
    write_req_entry_starts: torch.Tensor,
    write_req_entry_counts: torch.Tensor,
    write_req_active_mask: torch.Tensor,
    seed: int,
    violation_ring: torch.Tensor,
    violation_ring_valid: torch.Tensor,
    violation_write_index: torch.Tensor,
    first_violation: torch.Tensor,
    first_violation_set: torch.Tensor,
    is_errored: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    kernel_kind: int,
    real_kv_buf: torch.Tensor,
    real_kv_slot_stride_bytes: int,
    real_kv_read_bytes: int,
    real_kv_hash_mode: int,
) -> None:
    """Pure-torch / pure-Python reference implementation of :func:`canary_step`.

    Bit-wise equivalent of the CUDA kernel in
    ``jit_kernel/csrc/kv_cache_canary/canary.cuh`` for use as a
    differential-test oracle: write to it with the same arguments you
    would pass to :func:`canary_step` and the resulting tensor state must
    match byte-for-byte.

    Semantics replicated:

    - **Write path**: for every active write-req row, walk the splitmix64
      chain over ``[entry_starts, entry_starts + entry_counts)`` and
      store ``(req_id, token_id, position, prev_hash)`` per slot. Chain
      head is either ``seed`` (when ``seed_slot < 0``) or
      ``splitmix64_mix`` of the seed slot's stored
      ``(prev_hash, token, position)``.
    - **Verify path**: for every active verify entry, load the slot's
      stored fields plus the previous slot's fields, recompute
      ``expected_prev_hash`` via ``splitmix64_mix``, and compare three
      fail reasons in the same priority order as the kernel: ``REQ_ID`` >
      ``POSITION_MONOTONIC`` > ``HASH``.
    - **Violation ring + first-violation latch**: ``first_violation`` is
      latched once per launch (CAS-style); ring rows fill sequentially
      until capacity. The CUDA kernel uses atomic CAS, so multiple
      threads competing for the same ring row produce
      implementation-defined ordering — see the test scenarios for the
      contract this reference upholds (one violation per scenario, or
      independent violations across separate ring rows).
    - **Counters**: ``slot_run_counter += active_verify + active_writes``;
      ``kernel_run_counter += 1``.

    All tensor mutations happen in-place to match the kernel's
    out-parameter style; the function returns ``None``. Inputs may live
    on any device (CPU, CUDA, …); the reference does its work in Python
    after pulling the few small index tensors to host.
    """
    int_views = _IntViewBundle(
        src_buf=src_buf,
        dst_buf=dst_buf,
        slot_stride_bytes=slot_stride_bytes,
    )
    real_kv_view = _RealKvView(
        real_kv_buf=real_kv_buf,
        slot_stride_bytes=int(real_kv_slot_stride_bytes),
        read_bytes=int(real_kv_read_bytes),
        mode=int(real_kv_hash_mode),
    )
    plan = _RefPlan.from_tensors(
        verify_slot_indices=verify_slot_indices,
        verify_positions=verify_positions,
        verify_req_ids=verify_req_ids,
        verify_prev_slot_indices=verify_prev_slot_indices,
        verify_active_mask=verify_active_mask,
        write_slot_indices=write_slot_indices,
        write_token_ids=write_token_ids,
        write_positions=write_positions,
        write_req_ids=write_req_ids,
        write_req_seed_slot_indices=write_req_seed_slot_indices,
        write_req_entry_starts=write_req_entry_starts,
        write_req_entry_counts=write_req_entry_counts,
        write_req_active_mask=write_req_active_mask,
    )
    sink = _RefViolationSink(
        violation_ring=violation_ring,
        violation_ring_valid=violation_ring_valid,
        violation_write_index=violation_write_index,
        first_violation=first_violation,
        first_violation_set=first_violation_set,
        is_errored=is_errored,
        kernel_kind=kernel_kind,
    )

    active_verify = _run_verify_entries(
        plan=plan,
        int_views=int_views,
        real_kv_view=real_kv_view,
        seed=seed,
        sink=sink,
    )
    active_write = _run_write_chains(
        plan=plan,
        int_views=int_views,
        real_kv_view=real_kv_view,
        seed=seed,
    )

    # Commit any updated slot rows back to the destination buffer.
    int_views.commit()
    sink.commit()

    slot_run_counter.add_(active_verify + active_write)
    kernel_run_counter.add_(1)


class _IntViewBundle:
    """Host-side int64 view of src/dst slot byte buffers.

    The CUDA kernel reads/writes 64-bit fields directly out of byte
    buffers. We mirror that by pulling both buffers to host once,
    reshaping them as ``(num_slots, slot_stride_bytes // 8)`` int64
    arrays, and writing changes back on commit.
    """

    def __init__(
        self,
        *,
        src_buf: torch.Tensor,
        dst_buf: torch.Tensor,
        slot_stride_bytes: int,
    ) -> None:
        self._src_buf = src_buf
        self._dst_buf = dst_buf
        self._slot_stride_bytes = int(slot_stride_bytes)
        self._slot_stride_i64 = self._slot_stride_bytes // 8
        self._src_i64 = src_buf.detach().to("cpu").view(torch.int64).clone()
        if dst_buf.data_ptr() == src_buf.data_ptr():
            self._dst_i64 = self._src_i64
        else:
            self._dst_i64 = dst_buf.detach().to("cpu").view(torch.int64).clone()

    def load_field(self, slot_idx: int, field: int) -> int:
        row_start = slot_idx * self._slot_stride_i64
        return int(self._src_i64[row_start + field].item())

    def store_field(self, slot_idx: int, field: int, value: int) -> None:
        row_start = slot_idx * self._slot_stride_i64
        self._dst_i64[row_start + field] = to_signed_int64(value & _U64_MASK)

    def commit(self) -> None:
        """Write the (possibly modified) dst buffer back into the user tensor.

        The kernel only mutates ``dst_buf``; ``src_buf`` is read-only, so
        the non-aliased case leaves ``src_buf`` untouched on the device.
        When ``src_buf is dst_buf`` (aliased), the two host shadows share
        storage and the single copy below updates both.
        """
        dst_dev_i64 = self._dst_i64.to(self._dst_buf.device)
        self._dst_buf.view(torch.int64).copy_(dst_dev_i64)


class _RealKvView:
    """Host-side reader for the optional real-KV slot buffer.

    In ``OFF`` mode :meth:`hash_slot` always returns 0; in ``BIT`` /
    ``ALL`` modes it pulls ``read_bytes`` consecutive bytes starting at
    the slot's stride offset and folds them through splitmix64 in 8-byte
    chunks (zero-padded if ``read_bytes`` is not a multiple of 8). The
    CUDA kernel uses the same chunked-XOR mix so this implementation is
    a bit-exact reference.
    """

    def __init__(
        self,
        *,
        real_kv_buf: torch.Tensor,
        slot_stride_bytes: int,
        read_bytes: int,
        mode: int,
    ) -> None:
        self._mode = int(mode)
        self._slot_stride_bytes = int(slot_stride_bytes)
        self._read_bytes = int(read_bytes)
        if self._mode == REAL_KV_HASH_MODE_OFF or self._read_bytes <= 0:
            self._host_bytes: Optional[bytes] = None
        else:
            flat_bytes = real_kv_buf.detach().to("cpu").contiguous().view(torch.uint8)
            self._host_bytes = bytes(flat_bytes.numpy().tobytes())

    def hash_slot(self, slot_idx: int) -> int:
        """Return the splitmix64-folded hash of the real-KV slot, or 0 if disabled."""
        if self._mode == REAL_KV_HASH_MODE_OFF or self._host_bytes is None:
            return 0
        start = int(slot_idx) * self._slot_stride_bytes
        end = start + self._read_bytes
        if start < 0 or end > len(self._host_bytes):
            return 0
        chunk = self._host_bytes[start:end]
        acc = 0
        i = 0
        while i < self._read_bytes:
            j = min(i + 8, self._read_bytes)
            word_bytes = chunk[i:j]
            word = int.from_bytes(word_bytes, byteorder="little", signed=False)
            acc = splitmix64_mix(acc, word, 0)
            i = j
        return acc & _U64_MASK


class _RefPlan:
    """Host-side host-list view of the verify + write plan tensors."""

    def __init__(
        self,
        *,
        verify_slot_indices: list,
        verify_positions: list,
        verify_req_ids: list,
        verify_prev_slot_indices: list,
        verify_active_mask: list,
        write_slot_indices: list,
        write_token_ids: list,
        write_positions: list,
        write_req_ids: list,
        write_req_seed_slot_indices: list,
        write_req_entry_starts: list,
        write_req_entry_counts: list,
        write_req_active_mask: list,
    ) -> None:
        self.verify_slot_indices = verify_slot_indices
        self.verify_positions = verify_positions
        self.verify_req_ids = verify_req_ids
        self.verify_prev_slot_indices = verify_prev_slot_indices
        self.verify_active_mask = verify_active_mask
        self.write_slot_indices = write_slot_indices
        self.write_token_ids = write_token_ids
        self.write_positions = write_positions
        self.write_req_ids = write_req_ids
        self.write_req_seed_slot_indices = write_req_seed_slot_indices
        self.write_req_entry_starts = write_req_entry_starts
        self.write_req_entry_counts = write_req_entry_counts
        self.write_req_active_mask = write_req_active_mask

    @classmethod
    def from_tensors(cls, **tensors: torch.Tensor) -> "_RefPlan":
        as_list = {k: v.detach().to("cpu").tolist() for k, v in tensors.items()}
        return cls(**as_list)


class _RefViolationSink:
    """Host-side view of violation ring + first_violation + is_errored."""

    def __init__(
        self,
        *,
        violation_ring: torch.Tensor,
        violation_ring_valid: torch.Tensor,
        violation_write_index: torch.Tensor,
        first_violation: torch.Tensor,
        first_violation_set: torch.Tensor,
        is_errored: torch.Tensor,
        kernel_kind: int,
    ) -> None:
        self._violation_ring = violation_ring
        self._violation_ring_valid = violation_ring_valid
        self._violation_write_index = violation_write_index
        self._first_violation = first_violation
        self._first_violation_set = first_violation_set
        self._is_errored = is_errored
        self._kernel_kind = int(kernel_kind)

        self._ring_host = violation_ring.detach().to("cpu").clone()
        self._ring_valid_host = violation_ring_valid.detach().to("cpu").clone()
        self._write_index_host = int(violation_write_index.detach().to("cpu").item())
        self._first_violation_host = first_violation.detach().to("cpu").clone()
        self._first_violation_set_host = int(
            first_violation_set.detach().to("cpu").item()
        )
        self._is_errored_host = int(is_errored.detach().to("cpu").item())
        self._ring_capacity = int(self._ring_host.shape[0])

    def record(
        self,
        *,
        fail_reason: int,
        slot_idx: int,
        req_id: int,
        token_id: int,
        position: int,
        expected_hash: int,
        actual_hash: int,
    ) -> None:
        entry = [
            self._kernel_kind,
            int(fail_reason),
            int(slot_idx),
            int(req_id),
            int(token_id),
            int(position),
            to_signed_int64(expected_hash & _U64_MASK),
            to_signed_int64(actual_hash & _U64_MASK),
        ]

        if self._first_violation_set_host == 0:
            for i, value in enumerate(entry):
                self._first_violation_host[i] = value
            # Mirror the kernel's 2-stage CAS latch (claimed=1, committed=2).
            self._first_violation_set_host = 2

        seq = self._write_index_host
        self._write_index_host += 1
        idx = seq % self._ring_capacity
        if int(self._ring_valid_host[idx].item()) == 0:
            for i, value in enumerate(entry):
                self._ring_host[idx, i] = value
            self._ring_valid_host[idx] = 2

        self._is_errored_host = 1

    def commit(self) -> None:
        self._violation_ring.copy_(self._ring_host.to(self._violation_ring.device))
        self._violation_ring_valid.copy_(
            self._ring_valid_host.to(self._violation_ring_valid.device)
        )
        self._violation_write_index.fill_(self._write_index_host)
        self._first_violation.copy_(
            self._first_violation_host.to(self._first_violation.device)
        )
        self._first_violation_set.fill_(self._first_violation_set_host)
        self._is_errored.fill_(self._is_errored_host)


def _run_verify_entries(
    *,
    plan: _RefPlan,
    int_views: _IntViewBundle,
    real_kv_view: _RealKvView,
    seed: int,
    sink: _RefViolationSink,
) -> int:
    """Walk active verify entries, updating ``sink``. Returns active count."""
    active = 0
    n = len(plan.verify_active_mask)
    for tid in range(n):
        if int(plan.verify_active_mask[tid]) == 0:
            continue
        active += 1
        slot_idx = int(plan.verify_slot_indices[tid])
        expected_req_id = int(plan.verify_req_ids[tid])
        expected_position = int(plan.verify_positions[tid])
        prev_slot_idx = int(plan.verify_prev_slot_indices[tid])

        actual_req_id = int_views.load_field(slot_idx, _CANARY_FIELD_REQ_ID)
        actual_token_id = int_views.load_field(slot_idx, _CANARY_FIELD_TOKEN_ID)
        actual_position = int_views.load_field(slot_idx, _CANARY_FIELD_POSITION)
        actual_prev_hash = (
            int_views.load_field(slot_idx, _CANARY_FIELD_PREV_HASH) & _U64_MASK
        )
        actual_real_kv_hash = (
            int_views.load_field(slot_idx, _CANARY_FIELD_REAL_KV_HASH) & _U64_MASK
        )

        if prev_slot_idx < 0:
            expected_prev_hash = seed & _U64_MASK
        else:
            prev_prev_hash = (
                int_views.load_field(prev_slot_idx, _CANARY_FIELD_PREV_HASH) & _U64_MASK
            )
            prev_token = int_views.load_field(prev_slot_idx, _CANARY_FIELD_TOKEN_ID)
            prev_position = int_views.load_field(prev_slot_idx, _CANARY_FIELD_POSITION)
            expected_prev_hash = splitmix64_mix(
                prev_prev_hash, prev_token & _U64_MASK, prev_position & _U64_MASK
            )

        expected_real_kv_hash = real_kv_view.hash_slot(slot_idx)

        fail_reason = FailReason.NONE
        if actual_req_id != expected_req_id:
            fail_reason = FailReason.REQ_ID
        elif actual_position != expected_position:
            fail_reason = FailReason.POSITION_MONOTONIC
        elif actual_prev_hash != expected_prev_hash:
            fail_reason = FailReason.HASH
        elif actual_real_kv_hash != expected_real_kv_hash:
            fail_reason = FailReason.REAL_KV_HASH
        if fail_reason != FailReason.NONE:
            if fail_reason == FailReason.REAL_KV_HASH:
                expected_hash_field = expected_real_kv_hash
                actual_hash_field = actual_real_kv_hash
            else:
                expected_hash_field = expected_prev_hash
                actual_hash_field = actual_prev_hash
            sink.record(
                fail_reason=int(fail_reason),
                slot_idx=slot_idx,
                req_id=actual_req_id,
                token_id=actual_token_id,
                position=actual_position,
                expected_hash=expected_hash_field,
                actual_hash=actual_hash_field,
            )

    return active


def _run_write_chains(
    *,
    plan: _RefPlan,
    int_views: _IntViewBundle,
    real_kv_view: _RealKvView,
    seed: int,
) -> int:
    """Walk active write-req chains, mutating slots. Returns total slot count."""
    active_slots = 0
    n_reqs = len(plan.write_req_active_mask)
    for req_tid in range(n_reqs):
        if int(plan.write_req_active_mask[req_tid]) == 0:
            continue
        entry_start = int(plan.write_req_entry_starts[req_tid])
        entry_count = int(plan.write_req_entry_counts[req_tid])
        if entry_count <= 0:
            continue
        seed_slot_idx = int(plan.write_req_seed_slot_indices[req_tid])

        if seed_slot_idx < 0:
            prev_hash = seed & _U64_MASK
        else:
            seed_prev_hash = (
                int_views.load_field(seed_slot_idx, _CANARY_FIELD_PREV_HASH) & _U64_MASK
            )
            seed_token = int_views.load_field(seed_slot_idx, _CANARY_FIELD_TOKEN_ID)
            seed_position = int_views.load_field(seed_slot_idx, _CANARY_FIELD_POSITION)
            prev_hash = splitmix64_mix(
                seed_prev_hash,
                seed_token & _U64_MASK,
                seed_position & _U64_MASK,
            )

        for k in range(entry_count):
            i = entry_start + k
            slot_idx = int(plan.write_slot_indices[i])
            req_id = int(plan.write_req_ids[i])
            token_id = int(plan.write_token_ids[i])
            position = int(plan.write_positions[i])
            real_kv_hash = real_kv_view.hash_slot(slot_idx)

            int_views.store_field(slot_idx, _CANARY_FIELD_REQ_ID, req_id)
            int_views.store_field(slot_idx, _CANARY_FIELD_TOKEN_ID, token_id)
            int_views.store_field(slot_idx, _CANARY_FIELD_POSITION, position)
            int_views.store_field(slot_idx, _CANARY_FIELD_PREV_HASH, prev_hash)
            int_views.store_field(slot_idx, _CANARY_FIELD_REAL_KV_HASH, real_kv_hash)

            prev_hash = splitmix64_mix(
                prev_hash, token_id & _U64_MASK, position & _U64_MASK
            )
            active_slots += 1

    return active_slots
