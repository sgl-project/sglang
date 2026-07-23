from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.consts import splitmix64, splitmix64_mix3
from sglang.kernels.ops.kv_canary.verify import VerifyPlan
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.test.kernels.kv_canary._constants import (
    _I64_SIGN_BIT,
    _U64_MASK,
    DEFAULT_NUM_SLOTS,
    DEFAULT_RING_CAPACITY,
    DEFAULT_SLOT_STRIDE_BYTES,
)
from sglang.test.kernels.kv_canary._fixtures import (
    make_real_kv_source,
    make_real_kv_sources,
)

__all__ = [
    "FakeViolationLog",
    "assert_canary_buf_equal",
    "assert_canary_state_equal",
    "assert_only_bits_set",
    "chain_anchor_signed",
    "make_canary_buf",
    "make_canary_buf_pair",
    "make_log_pair",
    "make_real_kv_source",
    "make_real_kv_sources",
    "make_verify_plan",
    "make_verify_plan_pair",
    "make_write_plan",
    "make_write_plan_pair",
    "read_slot_fields",
    "stamp_clean_chain",
    "stamp_pair",
    "to_signed_int64",
    "write_slot_fields",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeViolationLog:
    ring: torch.Tensor
    write_index: torch.Tensor
    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor
    enable_chain_position_assert: torch.Tensor

    @classmethod
    def allocate(
        cls, *, capacity: int = DEFAULT_RING_CAPACITY, device: torch.device
    ) -> FakeViolationLog:
        return cls(
            ring=torch.zeros(
                capacity, consts.VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            write_index=torch.zeros(1, dtype=torch.int32, device=device),
            slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
            enable_chain_position_assert=torch.ones(
                1, dtype=torch.int32, device=device
            ),
        )


def make_canary_buf(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    slot_stride_bytes: int = DEFAULT_SLOT_STRIDE_BYTES,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device=device)


def make_canary_buf_pair(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    slot_stride_bytes: int = DEFAULT_SLOT_STRIDE_BYTES,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cuda_buf = make_canary_buf(
        num_slots=num_slots, slot_stride_bytes=slot_stride_bytes, device=device
    )
    return cuda_buf, cuda_buf.clone()


def make_log_pair(
    *,
    capacity: int = DEFAULT_RING_CAPACITY,
    device: torch.device,
) -> tuple[FakeViolationLog, FakeViolationLog]:
    return (
        FakeViolationLog.allocate(capacity=capacity, device=device),
        FakeViolationLog.allocate(capacity=capacity, device=device),
    )


def make_verify_plan(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    expected_input_ids: Optional[list[int]] = None,
    capacity: Optional[int] = None,
    device: torch.device,
) -> VerifyPlan:
    """Build a VerifyPlan whose active prefix matches the three input lists.

    Active prefix mirrors the input lists. Tail entries are left at the
    allocate-time defaults; ``verify_num_valid = len(slot_indices)``.

    ``expected_input_ids`` defaults to ``[-1] * n_active`` (the verify-kernel
    "skip token check" sentinel) so existing tests that only exercise the
    chain / position / real-kv-hash paths keep working unchanged.
    """
    n_active = len(slot_indices)
    if not (len(positions) == n_active and len(prev_slot_indices) == n_active):
        raise ValueError(
            "make_verify_plan: slot_indices, positions, and prev_slot_indices must all have the same length"
        )
    if expected_input_ids is None:
        expected_input_ids = [-1] * n_active
    if len(expected_input_ids) != n_active:
        raise ValueError(
            "make_verify_plan: expected_input_ids must match len(slot_indices)"
        )
    cap = capacity if capacity is not None else max(n_active, 1)
    plan = VerifyPlan.allocate(verify_capacity=cap, device=device)
    if n_active > 0:
        plan.verify_slot_indices[:n_active] = torch.tensor(
            slot_indices, dtype=torch.int64, device=device
        )
        plan.verify_expected_tokens[:n_active] = torch.tensor(
            expected_input_ids, dtype=torch.int64, device=device
        )
        plan.verify_expected_positions[:n_active] = torch.tensor(
            positions, dtype=torch.int64, device=device
        )
        plan.verify_prev_slot_indices[:n_active] = torch.tensor(
            prev_slot_indices, dtype=torch.int64, device=device
        )
    plan.verify_num_valid[0] = n_active
    return plan


def make_verify_plan_pair(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    expected_input_ids: Optional[list[int]] = None,
    capacity: Optional[int] = None,
    device: torch.device,
) -> tuple[VerifyPlan, VerifyPlan]:
    return (
        make_verify_plan(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=prev_slot_indices,
            expected_input_ids=expected_input_ids,
            capacity=capacity,
            device=device,
        ),
        make_verify_plan(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=prev_slot_indices,
            expected_input_ids=expected_input_ids,
            capacity=capacity,
            device=device,
        ),
    )


def make_write_plan(
    *,
    write_offsets: list[int],
    seed_slot_indices: list[int],
    num_valid_reqs: int,
    req_capacity: Optional[int] = None,
    device: torch.device,
) -> WritePlan:
    """Build a WritePlan from raw offsets and seed slot lists.

    ``write_offsets`` must have length ``len(seed_slot_indices) + 1`` (the trailing total entry count).
    """
    n_active = len(seed_slot_indices)
    if len(write_offsets) != n_active + 1:
        raise ValueError(
            "make_write_plan: write_offsets must have length len(seed_slot_indices) + 1"
        )
    cap = req_capacity if req_capacity is not None else max(n_active, 1)
    plan = WritePlan.allocate(write_req_capacity=cap, device=device)
    if n_active > 0:
        plan.write_seed_slot_indices[:n_active] = torch.tensor(
            seed_slot_indices, dtype=torch.int64, device=device
        )
    plan.write_offsets[: n_active + 1] = torch.tensor(
        write_offsets, dtype=torch.int64, device=device
    )
    plan.write_num_valid_reqs[0] = num_valid_reqs
    return plan


def make_write_plan_pair(
    *,
    write_offsets: list[int],
    seed_slot_indices: list[int],
    num_valid_reqs: int,
    req_capacity: Optional[int] = None,
    device: torch.device,
) -> tuple[WritePlan, WritePlan]:
    return (
        make_write_plan(
            write_offsets=write_offsets,
            seed_slot_indices=seed_slot_indices,
            num_valid_reqs=num_valid_reqs,
            req_capacity=req_capacity,
            device=device,
        ),
        make_write_plan(
            write_offsets=write_offsets,
            seed_slot_indices=seed_slot_indices,
            num_valid_reqs=num_valid_reqs,
            req_capacity=req_capacity,
            device=device,
        ),
    )


def to_signed_int64(value: int) -> int:
    value &= _U64_MASK
    if value >= _I64_SIGN_BIT:
        value -= 1 << 64
    return value


def chain_anchor_signed() -> int:
    return to_signed_int64(splitmix64(consts.CANARY_CHAIN_ANCHOR))


def write_slot_fields(
    *,
    canary_buf: torch.Tensor,
    slot_idx: int,
    token: int,
    position: int,
    prev_hash: int,
    real_kv_hash: int,
) -> None:
    view = canary_buf.view(torch.int64)
    view[slot_idx, 0] = token
    view[slot_idx, 1] = position
    view[slot_idx, 2] = prev_hash
    view[slot_idx, 3] = real_kv_hash


def stamp_pair(
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    *,
    slot_idx: int,
    token: int,
    position: int,
    prev_hash: int,
    real_kv_hash: int = 0,
) -> None:
    """Stamp the same slot fields into both (cuda, ref) canary buffers."""
    for buf in buf_pair:
        write_slot_fields(
            canary_buf=buf,
            slot_idx=slot_idx,
            token=token,
            position=position,
            prev_hash=prev_hash,
            real_kv_hash=real_kv_hash,
        )


def read_slot_fields(
    *, canary_buf: torch.Tensor, slot_idx: int
) -> tuple[int, int, int, int]:
    row = canary_buf.view(torch.int64)[slot_idx, :4].detach().cpu().tolist()
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


def stamp_clean_chain(
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    slot_indices: list[int],
    tokens: list[int],
    positions: list[int],
    real_kv_hashes: Optional[list[int]] = None,
) -> list[int]:
    n = len(tokens)
    real_kv_hashes = real_kv_hashes if real_kv_hashes is not None else [0] * n
    running_prev_hash = splitmix64(consts.CANARY_CHAIN_ANCHOR)
    stored_prev_hashes: list[int] = []
    for slot_idx, token, position, real_kv_hash in zip(
        slot_indices, tokens, positions, real_kv_hashes
    ):
        signed_prev = to_signed_int64(running_prev_hash)
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=signed_prev,
                real_kv_hash=to_signed_int64(real_kv_hash),
            )
        stored_prev_hashes.append(signed_prev)
        running_prev_hash = splitmix64_mix3(running_prev_hash, token, position)
    return stored_prev_hashes


def assert_canary_state_equal(
    *, log_a: FakeViolationLog, log_b: FakeViolationLog
) -> None:
    for name in ("ring", "write_index", "slot_run_counter", "kernel_run_counter"):
        assert torch.equal(
            getattr(log_a, name), getattr(log_b, name)
        ), f"{name} diverged (CUDA vs ref)"


def assert_canary_buf_equal(*, buf_a: torch.Tensor, buf_b: torch.Tensor) -> None:
    assert torch.equal(buf_a, buf_b), "canary_buf diverged (CUDA vs ref)"


def assert_only_bits_set(fail_bits: int, expected_bits: int) -> None:
    assert (
        fail_bits & expected_bits
    ) == expected_bits, (
        f"missing expected bits: expected {expected_bits:#b} got {fail_bits:#b}"
    )
    assert (
        fail_bits & ~expected_bits
    ) == 0, f"unexpected extra bits: got {fail_bits:#b} extras {fail_bits & ~expected_bits:#b}"
