"""Test-only Python reference for the canary kernel.

Two public surfaces:

- :func:`splitmix64_mix` — Python mirror of the device-side chain hash
  in ``jit_kernel/csrc/kv_cache_canary/canary.cuh``. Used by other
  canary tests as the oracle for "what hash should the kernel have
  computed". ``test_splitmix64_consistency.py`` cross-validates this
  mirror against the actual kernel to keep them bit-identical.
- :func:`canary_step_torch_reference` — pure-Python re-implementation
  of the full kernel (verify path + write-req chains + violation ring).
  Differential tests in ``test_kv_cache_canary.py`` drive both this and
  the CUDA kernel with the same arguments and require byte-equality.

Nothing in here is on the production hot path; production calls
:func:`sglang.jit_kernel.kv_cache_canary.canary_step` which dispatches
to the compiled CUDA kernel.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary import (
    _CANARY_FIELD_POSITION,
    _CANARY_FIELD_PREV_HASH,
    _CANARY_FIELD_REAL_KV_HASH,
    _CANARY_FIELD_TOKEN_ID,
    CANARY_EXPECTED_SKIP_SENTINEL,
    KERNEL_KIND_TAIL,
    REAL_KV_HASH_MODE_OFF,
    SKIP_CHAIN_SENTINEL,
    FailReason,
    to_signed_int64,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlanGpu

_U64_MASK = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Standard splitmix64 finalizer (no `+= GOLDEN` step).

    Matches the device-side ``splitmix64_finalize`` in ``canary.cuh``.
    """
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def splitmix64_mix(prev_hash: int, token_id: int, position: int) -> int:
    """3-arg mix: XOR three uint64s and run splitmix64.

    Bit-wise equivalent of the device-side ``splitmix64_mix`` in
    ``canary.cuh``. Used for real-KV byte folding (where the third arg is
    a zero pad). The canary slot chain advance uses :func:`splitmix64_mix4`.
    """
    assert (
        0 <= prev_hash <= _U64_MASK
    ), f"kv-canary: splitmix64_mix prev_hash {prev_hash:#x} out of uint64 range"
    combined = (prev_hash & _U64_MASK) ^ (token_id & _U64_MASK) ^ (position & _U64_MASK)
    return splitmix64(combined)


def splitmix64_mix4(
    prev_hash: int, token_id: int, position: int, real_kv_hash: int
) -> int:
    """4-arg chain step used by the canary slot chain.

    Folds the predecessor's full ``(prev_hash, token_id, position,
    real_kv_hash)`` tuple into the new ``prev_hash``. In real-KV OFF mode
    the predecessor's ``real_kv_hash`` is always 0, collapsing this to
    the same value :func:`splitmix64_mix` would produce. Bit-wise
    equivalent of the device-side ``splitmix64_mix4`` in ``canary.cuh``.
    """
    assert (
        0 <= prev_hash <= _U64_MASK
    ), f"kv-canary: splitmix64_mix4 prev_hash {prev_hash:#x} out of uint64 range"
    combined = (
        (prev_hash & _U64_MASK)
        ^ (token_id & _U64_MASK)
        ^ (position & _U64_MASK)
        ^ (real_kv_hash & _U64_MASK)
    )
    return splitmix64(combined)


# API source of truth: docstring of canary_step in sglang.jit_kernel.kv_cache_canary
def canary_step_torch_reference(
    *,
    buf: torch.Tensor,
    plan: BatchPlanGpu,
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
      store ``(token_id, position, prev_hash, real_kv_hash)`` per slot.
      Chain head is either ``seed`` (when ``seed_slot < 0``) or
      ``splitmix64_mix4`` of the seed slot's stored
      ``(prev_hash, token, position, real_kv_hash)``.
    - **Verify path**: for every active verify entry, load the slot's
      stored fields plus the previous slot's fields, recompute
      ``expected_prev_hash`` via ``splitmix64_mix4``, and compare fail
      reasons in the same priority order as the kernel:
      ``POSITION_MONOTONIC`` > ``HASH`` > ``REAL_KV_HASH``.
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
    int_views = _IntViewBundle(buf=buf)
    real_kv_view = _RealKvView(
        real_kv_buf=real_kv_buf,
        read_bytes=int(real_kv_read_bytes),
        mode=int(real_kv_hash_mode),
    )
    ref_plan = _RefPlan.from_batch_plan_gpu(plan)
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
        plan=ref_plan,
        int_views=int_views,
        real_kv_view=real_kv_view,
        seed=seed,
        sink=sink,
    )
    active_write = _run_write_chains(
        plan=ref_plan,
        int_views=int_views,
        real_kv_view=real_kv_view,
        seed=seed,
        sink=sink,
    )

    # Commit any updated slot rows back to the destination buffer.
    int_views.commit()
    sink.commit()

    slot_run_counter.add_(active_verify + active_write)
    kernel_run_counter.add_(1)


class _IntViewBundle:
    """Host-side int64 view of the canary slot byte buffer.

    The CUDA kernel reads/writes 64-bit fields directly out of the byte
    buffer. We mirror that by pulling the buffer to host once,
    reshaping it as ``(num_slots, slot_stride_bytes // 8)`` int64
    arrays, and writing changes back on commit. The per-slot byte
    stride is taken from ``buf.shape[1]``.
    """

    def __init__(self, *, buf: torch.Tensor) -> None:
        self._buf = buf
        self._slot_stride_bytes = int(buf.shape[1])
        self._slot_stride_i64 = self._slot_stride_bytes // 8
        self._buf_i64 = buf.detach().to("cpu").view(torch.int64).flatten().clone()

    def load_field(self, slot_idx: int, field: int) -> int:
        row_start = slot_idx * self._slot_stride_i64
        return int(self._buf_i64[row_start + field].item())

    def load_field_dst(self, slot_idx: int, field: int) -> int:
        # Single-buffer endpoint: dst and src are the same buffer.
        return self.load_field(slot_idx, field)

    def store_field(self, slot_idx: int, field: int, value: int) -> None:
        row_start = slot_idx * self._slot_stride_i64
        self._buf_i64[row_start + field] = to_signed_int64(value & _U64_MASK)

    def commit(self) -> None:
        """Write the (possibly modified) host buffer back into the user tensor."""
        dst_dev_i64 = self._buf_i64.to(self._buf.device)
        self._buf.view(torch.int64).copy_(
            dst_dev_i64.view(self._buf.shape[0], self._slot_stride_i64)
        )


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
        read_bytes: int,
        mode: int,
    ) -> None:
        self._mode = int(mode)
        self._slot_stride_bytes = int(real_kv_buf.shape[1])
        self._read_bytes = int(read_bytes)
        if self._mode == REAL_KV_HASH_MODE_OFF or self._read_bytes <= 0:
            self._host_bytes: Optional[bytes] = None
        else:
            flat_bytes = (
                real_kv_buf.detach().to("cpu").contiguous().view(torch.uint8).flatten()
            )
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
        verify_slot_indices: list[int],
        verify_positions: list[int],
        verify_prev_slot_indices: list[int],
        verify_num_valid: int,
        write_slot_indices: list[int],
        write_token_ids: list[int],
        write_positions: list[int],
        write_req_seed_slot_indices: list[int],
        write_req_entry_starts: list[int],
        write_req_entry_counts: list[int],
        write_req_num_valid: int,
        expected_write_token_ids: list[int],
        expected_write_positions: list[int],
    ) -> None:
        self.verify_slot_indices = verify_slot_indices
        self.verify_positions = verify_positions
        self.verify_prev_slot_indices = verify_prev_slot_indices
        self.verify_num_valid = verify_num_valid
        self.write_slot_indices = write_slot_indices
        self.write_token_ids = write_token_ids
        self.write_positions = write_positions
        self.write_req_seed_slot_indices = write_req_seed_slot_indices
        self.write_req_entry_starts = write_req_entry_starts
        self.write_req_entry_counts = write_req_entry_counts
        self.write_req_num_valid = write_req_num_valid
        self.expected_write_token_ids = expected_write_token_ids
        self.expected_write_positions = expected_write_positions

    @classmethod
    def from_batch_plan_gpu(cls, plan: BatchPlanGpu) -> "_RefPlan":
        return cls(
            verify_slot_indices=plan.verify_slot_indices.detach().to("cpu").tolist(),
            verify_positions=plan.verify_positions.detach().to("cpu").tolist(),
            verify_prev_slot_indices=plan.verify_prev_slot_indices.detach()
            .to("cpu")
            .tolist(),
            verify_num_valid=int(plan.verify_num_valid.detach().to("cpu").item()),
            write_slot_indices=plan.write_slot_indices.detach().to("cpu").tolist(),
            write_token_ids=plan.write_token_ids.detach().to("cpu").tolist(),
            write_positions=plan.write_positions.detach().to("cpu").tolist(),
            write_req_seed_slot_indices=plan.write_req_seed_slot_indices.detach()
            .to("cpu")
            .tolist(),
            write_req_entry_starts=plan.write_req_entry_starts.detach()
            .to("cpu")
            .tolist(),
            write_req_entry_counts=plan.write_req_entry_counts.detach()
            .to("cpu")
            .tolist(),
            write_req_num_valid=int(plan.write_req_num_valid.detach().to("cpu").item()),
            expected_write_token_ids=plan.expected_write_token_ids.detach()
            .to("cpu")
            .tolist(),
            expected_write_positions=plan.expected_write_positions.detach()
            .to("cpu")
            .tolist(),
        )


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
        token_id: int,
        position: int,
        expected_hash: int,
        actual_hash: int,
        expected_position: int,
    ) -> None:
        entry = [
            self._kernel_kind,
            int(fail_reason),
            int(slot_idx),
            int(token_id),
            int(position),
            to_signed_int64(expected_hash & _U64_MASK),
            to_signed_int64(actual_hash & _U64_MASK),
            int(expected_position),
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
    n = len(plan.verify_slot_indices)
    for tid in range(n):
        if tid >= plan.verify_num_valid:
            continue
        active += 1
        slot_idx = int(plan.verify_slot_indices[tid])
        expected_position = int(plan.verify_positions[tid])
        prev_slot_idx = int(plan.verify_prev_slot_indices[tid])

        actual_token_id = int_views.load_field(slot_idx, _CANARY_FIELD_TOKEN_ID)
        actual_position = int_views.load_field(slot_idx, _CANARY_FIELD_POSITION)
        actual_prev_hash = (
            int_views.load_field(slot_idx, _CANARY_FIELD_PREV_HASH) & _U64_MASK
        )
        actual_real_kv_hash = (
            int_views.load_field(slot_idx, _CANARY_FIELD_REAL_KV_HASH) & _U64_MASK
        )

        skip_chain_check = prev_slot_idx == SKIP_CHAIN_SENTINEL
        if skip_chain_check:
            expected_prev_hash = 0
        elif prev_slot_idx < 0:
            expected_prev_hash = seed & _U64_MASK
        else:
            prev_prev_hash = (
                int_views.load_field(prev_slot_idx, _CANARY_FIELD_PREV_HASH) & _U64_MASK
            )
            prev_token = int_views.load_field(prev_slot_idx, _CANARY_FIELD_TOKEN_ID)
            prev_position = int_views.load_field(prev_slot_idx, _CANARY_FIELD_POSITION)
            prev_real_kv_hash = (
                int_views.load_field(prev_slot_idx, _CANARY_FIELD_REAL_KV_HASH)
                & _U64_MASK
            )
            expected_prev_hash = splitmix64_mix4(
                prev_prev_hash,
                prev_token & _U64_MASK,
                prev_position & _U64_MASK,
                prev_real_kv_hash,
            )

        expected_real_kv_hash = real_kv_view.hash_slot(slot_idx)

        # Tail's src is head_shadow, which stores pre-model-write real_kv_hash
        # (head fires before the model writes). Skip the external real_kv_hash
        # check on the tail path — head and sweep (both reading tail_shadow =
        # post-write hash) cover this fail reason.
        real_kv_check_enabled = sink._kernel_kind != KERNEL_KIND_TAIL

        fail_reason = FailReason.NONE
        if actual_position != expected_position:
            fail_reason = FailReason.POSITION_MONOTONIC
        elif not skip_chain_check and actual_prev_hash != expected_prev_hash:
            fail_reason = FailReason.HASH
        elif real_kv_check_enabled and actual_real_kv_hash != expected_real_kv_hash:
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
                token_id=actual_token_id,
                position=actual_position,
                expected_hash=expected_hash_field,
                actual_hash=actual_hash_field,
                expected_position=expected_position,
            )

    return active


def _run_write_chains(
    *,
    plan: _RefPlan,
    int_views: _IntViewBundle,
    real_kv_view: _RealKvView,
    seed: int,
    sink: _RefViolationSink,
) -> int:
    """Walk active write-req chains, mutating slots. Returns total slot count."""
    active_slots = 0
    n_reqs = len(plan.write_req_seed_slot_indices)
    for req_tid in range(n_reqs):
        if req_tid >= plan.write_req_num_valid:
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
                int_views.load_field_dst(seed_slot_idx, _CANARY_FIELD_PREV_HASH)
                & _U64_MASK
            )
            seed_token = int_views.load_field_dst(seed_slot_idx, _CANARY_FIELD_TOKEN_ID)
            seed_position = int_views.load_field_dst(
                seed_slot_idx, _CANARY_FIELD_POSITION
            )
            seed_real_kv_hash = (
                int_views.load_field_dst(seed_slot_idx, _CANARY_FIELD_REAL_KV_HASH)
                & _U64_MASK
            )
            prev_hash = splitmix64_mix4(
                seed_prev_hash,
                seed_token & _U64_MASK,
                seed_position & _U64_MASK,
                seed_real_kv_hash,
            )

        for k in range(entry_count):
            i = entry_start + k
            slot_idx = int(plan.write_slot_indices[i])
            token_id = int(plan.write_token_ids[i])
            position = int(plan.write_positions[i])

            expected_token = int(plan.expected_write_token_ids[i])
            if (
                expected_token != CANARY_EXPECTED_SKIP_SENTINEL
                and expected_token != token_id
            ):
                sink.record(
                    fail_reason=int(FailReason.INPUT_TOKEN_MISMATCH),
                    slot_idx=slot_idx,
                    token_id=token_id,
                    position=position,
                    expected_hash=expected_token & _U64_MASK,
                    actual_hash=token_id & _U64_MASK,
                    expected_position=0,
                )
            expected_pos = int(plan.expected_write_positions[i])
            if (
                expected_pos != CANARY_EXPECTED_SKIP_SENTINEL
                and expected_pos != position
            ):
                sink.record(
                    fail_reason=int(FailReason.INPUT_POSITION_MISMATCH),
                    slot_idx=slot_idx,
                    token_id=token_id,
                    position=position,
                    expected_hash=0,
                    actual_hash=0,
                    expected_position=expected_pos,
                )

            real_kv_hash = real_kv_view.hash_slot(slot_idx)

            int_views.store_field(slot_idx, _CANARY_FIELD_TOKEN_ID, token_id)
            int_views.store_field(slot_idx, _CANARY_FIELD_POSITION, position)
            int_views.store_field(slot_idx, _CANARY_FIELD_PREV_HASH, prev_hash)
            int_views.store_field(slot_idx, _CANARY_FIELD_REAL_KV_HASH, real_kv_hash)

            prev_hash = splitmix64_mix4(
                prev_hash,
                token_id & _U64_MASK,
                position & _U64_MASK,
                real_kv_hash & _U64_MASK,
            )
            active_slots += 1

    return active_slots
