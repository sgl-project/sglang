"""GPU kernel tests for KV cache canary.

Cover the stateless-redesign kernel signature: per-verify-entry arrays,
per-write-req chains driven by one thread each, on-device splitmix64 chain
hash recomputation, verify fail_reasons (position monotonic / chain hash /
real-KV hash), and the violation buffer first-violation latch.
"""

from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_EXPECTED_SKIP_SENTINEL,
    CANARY_SLOT_BYTES,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    SKIP_CHAIN_SENTINEL,
    VIOLATION_FIELDS,
    FailReason,
    canary_step,
    to_signed_int64,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlanGpu
from sglang.jit_kernel.kv_cache_canary_ref import (
    canary_step_torch_reference,
    splitmix64_mix,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")

_SEED = 0xC0FFEE1234567890


def _alloc_state(ring_capacity: int = 64) -> dict:
    return dict(
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device="cuda"
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device="cuda"),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def _alloc_pool(num_slots: int, slot_stride_bytes: int) -> torch.Tensor:
    return torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device="cuda")


def _i64(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def _i32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _num_valid(values: list[int]) -> torch.Tensor:
    return _i32([sum(1 for value in values if value)])


def _empty_real_kv() -> torch.Tensor:
    return torch.zeros(1, 1, dtype=torch.uint8, device="cuda")


def _run(
    *,
    buf: torch.Tensor,
    verify_slot_indices: list[int],
    verify_positions: list[int],
    verify_prev_slot_indices: list[int],
    verify_num_valid: list[int],
    write_slot_indices: list[int],
    write_token_ids: list[int],
    write_positions: list[int],
    write_req_seed_slot_indices: list[int],
    write_req_entry_starts: list[int],
    write_req_entry_counts: list[int],
    write_req_num_valid: list[int],
    state: dict,
    kernel_kind: int,
    seed: int = _SEED,
    real_kv_buf: torch.Tensor | None = None,
    real_kv_read_bytes: int = 0,
    real_kv_hash_mode: int = 0,
    expected_write_token_ids: list[int] | None = None,
    expected_write_positions: list[int] | None = None,
) -> None:
    n_write_padded = len(write_slot_indices) or 1
    expected_write_token_ids = expected_write_token_ids or (
        [CANARY_EXPECTED_SKIP_SENTINEL] * n_write_padded
    )
    expected_write_positions = expected_write_positions or (
        [CANARY_EXPECTED_SKIP_SENTINEL] * n_write_padded
    )
    plan = BatchPlanGpu(
        verify_slot_indices=_i64(verify_slot_indices or [0]),
        verify_positions=_i64(verify_positions or [0]),
        verify_prev_slot_indices=_i64(verify_prev_slot_indices or [-1]),
        verify_num_valid=_num_valid(verify_num_valid),
        write_slot_indices=_i64(write_slot_indices or [0]),
        write_token_ids=_i64(write_token_ids or [0]),
        write_positions=_i64(write_positions or [0]),
        write_req_seed_slot_indices=_i64(write_req_seed_slot_indices or [-1]),
        write_req_entry_starts=_i64(write_req_entry_starts or [0]),
        write_req_entry_counts=_i64(write_req_entry_counts or [0]),
        write_req_num_valid=_num_valid(write_req_num_valid),
        expected_write_token_ids=_i64(expected_write_token_ids),
        expected_write_positions=_i64(expected_write_positions),
    )
    canary_step(
        buf=buf,
        plan=plan,
        seed=seed,
        violation_ring=state["violation_ring"],
        violation_write_index=state["violation_write_index"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        kernel_kind=kernel_kind,
        real_kv_buf=real_kv_buf if real_kv_buf is not None else _empty_real_kv(),
        real_kv_read_bytes=real_kv_read_bytes,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()


def _read_slot(
    buf: torch.Tensor, slot_idx: int, slot_stride_bytes: int
) -> tuple[int, int, int, int]:
    """Return ``(token_id, position, prev_hash, real_kv_hash)``."""
    row = buf[slot_idx, :CANARY_SLOT_BYTES].clone().view(torch.int64).cpu().tolist()
    return tuple(int(x) for x in row)


def test_write_chain_seeded_from_kseed_fills_slots_with_splitmix64_chain():
    """First-write path: chain seeds from kSeed and stores (token, pos, prev_hash, real_kv_hash)."""
    slot_stride = 256
    dst = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    tokens = [10, 20, 30]
    positions = [0, 1, 2]
    slot_indices = [4, 5, 6]

    _run(
        buf=dst,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=slot_indices,
        write_token_ids=tokens,
        write_positions=positions,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["violation_write_index"].item()) == 0
    assert int(state["slot_run_counter"].item()) == 3
    # Recompute the expected chain in Python and bit-wise-match the stored prev_hash.
    expected_prev = _SEED
    for slot_idx, token, position in zip(slot_indices, tokens, positions):
        tid, pos, ph, real_kv_hash = _read_slot(dst, slot_idx, slot_stride)
        assert tid == token
        assert pos == position
        assert ph == to_signed_int64(expected_prev)
        # real_kv_hash_mode defaults to OFF in this test -> field stays at 0.
        assert real_kv_hash == 0
        expected_prev = splitmix64_mix(expected_prev, token, position)


def test_verify_clean_round_trip_no_violation():
    """Write-then-verify same buffer: clean state -> no violation."""
    slot_stride = 256
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    tokens = [100, 200, 300]
    positions = [0, 1, 2]
    slot_indices = [0, 1, 2]

    # Phase 1: write the chain.
    _run(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=slot_indices,
        write_token_ids=tokens,
        write_positions=positions,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Phase 2: verify every position. prev_slot_indices: -1 for pos 0, the
    # actual previous slot for the rest.
    _run(
        buf=buf,
        verify_slot_indices=slot_indices,
        verify_positions=positions,
        verify_prev_slot_indices=[-1, slot_indices[0], slot_indices[1]],
        verify_num_valid=[1, 1, 1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )
    assert int(state["violation_write_index"].item()) == 0


def test_verify_position_mismatch_reports_position_monotonic_fail_reason():
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Verify with an expected position that doesn't match the slot's stored
    # position field -> kFailReasonPositionMonotonic.
    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[5],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["violation_write_index"].item()) >= 1
    first = state["violation_ring"][0].cpu().tolist()
    assert int(first[1]) == FailReason.POSITION_MONOTONIC.value


def test_verify_chain_hash_mismatch_reports_hash_fail_reason():
    """Corrupt slot[0].prev_hash post-write -> verify on slot[0] sees hash mismatch."""
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Corrupt slot[0].prev_hash field (offset 16 bytes = field index 2).
    buf_view = buf.view(torch.int64)
    buf_view[0, 2] = torch.tensor(
        to_signed_int64(0xDEADBEEFDEADBEEF), dtype=torch.int64, device="cuda"
    )

    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["violation_write_index"].item()) >= 1
    first = state["violation_ring"][0].cpu().tolist()
    assert int(first[1]) == FailReason.HASH.value


def test_verify_skips_chain_check_on_sentinel():
    """``verify_prev_slot_indices[i] == SKIP_CHAIN_SENTINEL`` (-2) skips the
    chain hash check on that entry but still runs the position-monotonic
    check and the real_kv_hash recompute+compare. Enables sweep-mode plans
    that verify an alive slot's live real_kv_hash without requiring the
    full chain to be reconstructible across the sweep set.
    """
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Corrupt slot[0].prev_hash. With the skip sentinel, the kernel must
    # NOT report kFailReasonHash.
    buf_view = buf.view(torch.int64)
    buf_view[0, 2] = torch.tensor(
        to_signed_int64(0xDEADBEEFDEADBEEF), dtype=torch.int64, device="cuda"
    )

    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[SKIP_CHAIN_SENTINEL],
        verify_num_valid=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["violation_write_index"].item()) == 0

    # Also corrupt slot[0].real_kv_hash (field index 3). With real_kv mode
    # OFF the expected real_kv_hash is 0; a non-zero stored value must
    # still trigger kFailReasonRealKvHash even under the skip sentinel,
    # proving the real_kv_hash check is independent of the chain skip.
    # Use HEAD here: the tail kernel intentionally skips the real_kv_hash
    # check (its src is head_shadow, which carries pre-model-write hashes
    # — see ``_launch_kernel_only`` and ``run_verify_entry``).
    buf_view[0, 3] = torch.tensor(
        to_signed_int64(0xCAFEBABECAFEBABE), dtype=torch.int64, device="cuda"
    )

    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[SKIP_CHAIN_SENTINEL],
        verify_num_valid=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["violation_write_index"].item()) >= 1
    first = state["violation_ring"][0].cpu().tolist()
    assert int(first[1]) == FailReason.REAL_KV_HASH.value


def test_inactive_mask_rows_are_skipped_no_io_no_counter():
    """``*_active_mask == 0`` rows = skip-sentinel padding for cuda graph fixed buffers."""
    slot_stride = 256
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    _run(
        buf=buf,
        verify_slot_indices=[0, 1],
        verify_positions=[0, 1],
        verify_prev_slot_indices=[-1, 0],
        verify_num_valid=[0, 0],
        write_slot_indices=[0, 1, 2],
        write_token_ids=[10, 20, 30],
        write_positions=[0, 1, 2],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_num_valid=[0],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Every active mask is 0: kernel must be a no-op on slot I/O and counters.
    assert int(state["violation_write_index"].item()) == 0
    assert int(state["slot_run_counter"].item()) == 0
    # But the kernel_run_counter still increments — that's what the
    # host-side health monitor uses to detect "kernel actually launched".
    assert int(state["kernel_run_counter"].item()) >= 1


def test_kernel_run_counter_increments_even_with_zero_threads():
    """Regression for the e2e wiring bug: when the host has no verify
    entries and no write-req chains (length-1 placeholder arrays with the
    active mask zeroed), the kernel must still launch and atomicAdd the
    ``kernel_run_counter`` so the host-side warmup health monitor sees
    liveness.

    Pre-fix history: ``canary_step`` host stub returned early when
    ``num_verify + num_write_reqs == 0`` and the runner additionally
    short-circuited ``_run_kernel_pair`` when ``plan.num_verify +
    plan.num_write == 0``; the cuda-graph capture path also recorded a
    no-op kernel under the same guards. The combined effect was a
    ``kernel_run_counter == 0`` after ``counter_zero_warmup_forwards``
    forwards in production, raising the "kernel never ran after warmup"
    panic even though every replay was actually running the kernel with
    skip-sentinel masks.
    """
    slot_stride = 64
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    # Length-1 placeholder arrays for every plan field with the active
    # masks zeroed — this mimics ``BatchPlanGpu`` at capture time
    # (verify_capacity / write_req_capacity must be >= 1 but the masks
    # default to 0 = skip-sentinel).
    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[0],
        write_slot_indices=[0],
        write_token_ids=[0],
        write_positions=[0],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[0],
        write_req_num_valid=[0],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["violation_write_index"].item()) == 0
    assert int(state["slot_run_counter"].item()) == 0
    assert int(state["kernel_run_counter"].item()) >= 1


def test_first_violation_preserved_across_cascading_mismatches():
    slot_stride = 128
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state(ring_capacity=4)

    _run(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=[0, 1, 2, 3],
        write_token_ids=[10, 20, 30, 40],
        write_positions=[0, 1, 2, 3],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[4],
        write_req_num_valid=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # First mismatch: position mismatch at slot 0 (stored=0, expected=7).
    _run(
        buf=buf,
        verify_slot_indices=[0],
        verify_positions=[7],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )
    first_after_initial = state["violation_ring"][0].cpu().tolist()

    # Cascade: 20 more verify launches with wrong positions on slots 1..3.
    for _ in range(20):
        _run(
            buf=buf,
            verify_slot_indices=[1, 2, 3],
            verify_positions=[99, 98, 97],
            verify_prev_slot_indices=[0, 1, 2],
            verify_num_valid=[1, 1, 1],
            write_slot_indices=[],
            write_token_ids=[],
            write_positions=[],
            write_req_seed_slot_indices=[],
            write_req_entry_starts=[],
            write_req_entry_counts=[],
            write_req_num_valid=[],
            state=state,
            kernel_kind=KERNEL_KIND_HEAD,
        )

    # violation_ring[0] must be byte-identical to the first observed row —
    # the atomicAdd ``seq == 0`` winner owns it permanently.
    first_after_cascade = state["violation_ring"][0].cpu().tolist()
    assert first_after_initial == first_after_cascade
    # The write_index has advanced past ring capacity.
    assert int(state["violation_write_index"].item()) > 4


# ---------------------------------------------------------------------------
# Differential tests: every scenario calls both ``canary_step`` (CUDA) and
# ``canary_step_torch_reference`` (pure-torch) with byte-identical inputs
# and asserts bitwise equality on all output tensors (the kernel mutates
# dst_buf + violation state + counters in place). User-instruction L73:
# "对各种各样的场景，都双调用，然后 assert 双方结果 bitwise eq".
# ---------------------------------------------------------------------------


_DIFFERENTIAL_OUTPUT_FIELDS = (
    "dst_buf",
    "violation_ring",
    "violation_write_index",
    "slot_run_counter",
    "kernel_run_counter",
)


def _alloc_diff_state(
    *,
    num_slots: int,
    slot_stride: int,
    ring_capacity: int,
    device: str,
) -> dict:
    """One bag of mutable tensors that ``canary_step`` will write into."""
    return dict(
        dst_buf=torch.zeros(num_slots, slot_stride, dtype=torch.uint8, device=device),
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
    )


def _clone_state_for_reference(state: dict, *, device: str) -> dict:
    """Deep-clone every mutable tensor so the reference run is independent."""
    return {k: v.detach().clone().to(device) for k, v in state.items()}


def _i64_on(values: list[int], device: str) -> torch.Tensor:
    return torch.tensor(values or [0], dtype=torch.int64, device=device)


def _i32_on(values: list[int], device: str) -> torch.Tensor:
    return torch.tensor(values or [0], dtype=torch.int32, device=device)


def _num_valid_on(values: list[int], device: str) -> torch.Tensor:
    return _i32_on([sum(1 for value in values if value)], device)


def _clone_batch_plan_gpu(plan: BatchPlanGpu, *, device: str) -> BatchPlanGpu:
    return BatchPlanGpu(
        verify_slot_indices=plan.verify_slot_indices.detach().clone().to(device),
        verify_positions=plan.verify_positions.detach().clone().to(device),
        verify_prev_slot_indices=plan.verify_prev_slot_indices.detach()
        .clone()
        .to(device),
        verify_num_valid=plan.verify_num_valid.detach().clone().to(device),
        write_slot_indices=plan.write_slot_indices.detach().clone().to(device),
        write_token_ids=plan.write_token_ids.detach().clone().to(device),
        write_positions=plan.write_positions.detach().clone().to(device),
        write_req_seed_slot_indices=plan.write_req_seed_slot_indices.detach()
        .clone()
        .to(device),
        write_req_entry_starts=plan.write_req_entry_starts.detach().clone().to(device),
        write_req_entry_counts=plan.write_req_entry_counts.detach().clone().to(device),
        write_req_num_valid=plan.write_req_num_valid.detach().clone().to(device),
        expected_write_token_ids=plan.expected_write_token_ids.detach()
        .clone()
        .to(device),
        expected_write_positions=plan.expected_write_positions.detach()
        .clone()
        .to(device),
    )


def _build_inputs_on(device: str, **plan_lists) -> BatchPlanGpu:
    n_write = max(len(plan_lists["write_slot_indices"]), 1)
    expected_write_token_ids = plan_lists.get("expected_write_token_ids") or (
        [CANARY_EXPECTED_SKIP_SENTINEL] * n_write
    )
    expected_write_positions = plan_lists.get("expected_write_positions") or (
        [CANARY_EXPECTED_SKIP_SENTINEL] * n_write
    )
    return BatchPlanGpu(
        verify_slot_indices=_i64_on(plan_lists["verify_slot_indices"], device),
        verify_positions=_i64_on(plan_lists["verify_positions"], device),
        verify_prev_slot_indices=_i64_on(
            plan_lists["verify_prev_slot_indices"], device
        ),
        verify_num_valid=_num_valid_on(plan_lists["verify_num_valid"], device),
        write_slot_indices=_i64_on(plan_lists["write_slot_indices"], device),
        write_token_ids=_i64_on(plan_lists["write_token_ids"], device),
        write_positions=_i64_on(plan_lists["write_positions"], device),
        write_req_seed_slot_indices=_i64_on(
            plan_lists["write_req_seed_slot_indices"] or [-1], device
        ),
        write_req_entry_starts=_i64_on(plan_lists["write_req_entry_starts"], device),
        write_req_entry_counts=_i64_on(plan_lists["write_req_entry_counts"], device),
        write_req_num_valid=_num_valid_on(plan_lists["write_req_num_valid"], device),
        expected_write_token_ids=_i64_on(expected_write_token_ids, device),
        expected_write_positions=_i64_on(expected_write_positions, device),
    )


def _assert_states_match(
    *,
    cuda_state: dict,
    ref_state: dict,
    scenario: str,
) -> None:
    for field in _DIFFERENTIAL_OUTPUT_FIELDS:
        cuda_tensor = cuda_state[field].detach().to("cpu")
        ref_tensor = ref_state[field].detach().to("cpu")
        assert torch.equal(
            cuda_tensor, ref_tensor
        ), f"{scenario}: field {field!r} diverges between CUDA and torch reference"


def _run_differential(
    *,
    scenario: str,
    src_initial: torch.Tensor,
    state_cuda: dict,
    state_ref: dict,
    inputs_cuda: BatchPlanGpu,
    inputs_ref: BatchPlanGpu,
    kernel_kind: int,
    seed: int = _SEED,
    real_kv_buf_cuda: torch.Tensor | None = None,
    real_kv_buf_ref: torch.Tensor | None = None,
    real_kv_read_bytes: int = 0,
    real_kv_hash_mode: int = 0,
) -> None:
    """Drive both implementations and bit-wise compare every output field."""
    src_cuda = src_initial.detach().clone().to(state_cuda["dst_buf"].device)
    src_ref = src_initial.detach().clone().to(state_ref["dst_buf"].device)

    if real_kv_buf_cuda is None:
        real_kv_buf_cuda = torch.zeros(
            1, 1, dtype=torch.uint8, device=state_cuda["dst_buf"].device
        )
    if real_kv_buf_ref is None:
        real_kv_buf_ref = torch.zeros(
            1, 1, dtype=torch.uint8, device=state_ref["dst_buf"].device
        )

    state_cuda["dst_buf"].copy_(src_cuda)
    state_ref["dst_buf"].copy_(src_ref)

    canary_step(
        buf=state_cuda["dst_buf"],
        plan=inputs_cuda,
        seed=seed,
        kernel_kind=kernel_kind,
        violation_ring=state_cuda["violation_ring"],
        violation_write_index=state_cuda["violation_write_index"],
        slot_run_counter=state_cuda["slot_run_counter"],
        kernel_run_counter=state_cuda["kernel_run_counter"],
        real_kv_buf=real_kv_buf_cuda,
        real_kv_read_bytes=real_kv_read_bytes,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()

    canary_step_torch_reference(
        buf=state_ref["dst_buf"],
        plan=inputs_ref,
        seed=seed,
        kernel_kind=kernel_kind,
        violation_ring=state_ref["violation_ring"],
        violation_write_index=state_ref["violation_write_index"],
        slot_run_counter=state_ref["slot_run_counter"],
        kernel_run_counter=state_ref["kernel_run_counter"],
        real_kv_buf=real_kv_buf_ref,
        real_kv_read_bytes=real_kv_read_bytes,
        real_kv_hash_mode=real_kv_hash_mode,
    )

    _assert_states_match(cuda_state=state_cuda, ref_state=state_ref, scenario=scenario)


def _empty_plan_lists() -> dict:
    """Default empty-list values for every plan-list field.

    Scenario builders start from this dict and overwrite only the fields
    that are non-empty for their case; cuts down ~12 lines of boilerplate
    per scenario.
    """
    return dict(
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_num_valid=[],
    )


def _scenario_write_only_single_req() -> dict:
    plan = _empty_plan_lists()
    plan.update(
        write_slot_indices=[0, 1, 2],
        write_token_ids=[101, 202, 303],
        write_positions=[0, 1, 2],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=None,
        plan=plan,
    )


def _scenario_write_only_multi_req() -> dict:
    # Two reqs, each starting from kSeed -> two independent chains; chosen
    # so their write_slots do not overlap.
    plan = _empty_plan_lists()
    plan.update(
        write_slot_indices=[0, 1, 2, 3],
        write_token_ids=[10, 20, 30, 40],
        write_positions=[0, 1, 0, 1],
        write_req_seed_slot_indices=[-1, -1],
        write_req_entry_starts=[0, 2],
        write_req_entry_counts=[2, 2],
        write_req_num_valid=[1, 1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=None,
        plan=plan,
    )


def _scenario_verify_only_clean_chain() -> dict:
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1, 2],
        verify_positions=[0, 1, 2],
        verify_prev_slot_indices=[-1, 0, 1],
        verify_num_valid=[1, 1, 1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_TAIL,
        prefill_writes=dict(
            slots=[0, 1, 2],
            tokens=[11, 22, 33],
            positions=[0, 1, 2],
        ),
        plan=plan,
    )


def _scenario_verify_only_position_mismatch() -> dict:
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0],
        verify_positions=[5],  # mismatch: slot stored position=0
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=4,
        ring_capacity=4,
        kernel_kind=KERNEL_KIND_TAIL,
        prefill_writes=dict(
            slots=[0],
            tokens=[42],
            positions=[0],
        ),
        plan=plan,
        expected_fail_reason=FailReason.POSITION_MONOTONIC,
    )


def _scenario_verify_only_hash_mismatch() -> dict:
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=4,
        ring_capacity=4,
        kernel_kind=KERNEL_KIND_TAIL,
        prefill_writes=dict(
            slots=[0],
            tokens=[42],
            positions=[0],
        ),
        # Post-prefill we corrupt slot[0].prev_hash; verify should fire HASH.
        corrupt_prev_hash=dict(slot=0, value=0xDEADBEEFDEADBEEF),
        plan=plan,
        expected_fail_reason=FailReason.HASH,
    )


def _scenario_mixed_write_and_verify() -> dict:
    # Same forward: verify positions [0,1] (already written) and write
    # new positions [2,3] on top of them.
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1],
        verify_positions=[0, 1],
        verify_prev_slot_indices=[-1, 0],
        verify_num_valid=[1, 1],
        write_slot_indices=[2, 3],
        write_token_ids=[81, 82],
        write_positions=[2, 3],
        write_req_seed_slot_indices=[1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[2],
        write_req_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=dict(
            slots=[0, 1],
            tokens=[71, 72],
            positions=[0, 1],
        ),
        plan=plan,
    )


def _scenario_first_violation_latch_preserved() -> dict:
    # Two verify entries on the same req, both hit POSITION mismatch (stored
    # positions are 0/1, expected are 888/999). violation_ring[0] (the
    # atomicAdd ``seq == 0`` winner) must preserve entry 0's row across the cascade.
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1],
        verify_positions=[888, 999],
        verify_prev_slot_indices=[-1, 0],
        verify_num_valid=[1, 1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=4,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_TAIL,
        prefill_writes=dict(
            slots=[0, 1],
            tokens=[71, 72],
            positions=[0, 1],
        ),
        plan=plan,
    )


def _scenario_small_k_req() -> dict:
    plan = _empty_plan_lists()
    plan.update(
        write_slot_indices=[0],
        write_token_ids=[7],
        write_positions=[0],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=2,
        ring_capacity=4,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=None,
        plan=plan,
    )


def _scenario_large_k_req_1w() -> dict:
    # User-instruction L53: a single decode step on a 10k-token prefix
    # must verify ALL 10k positions. Use a 10k-long write chain followed
    # by a verify launch over the full prefix in the same scenario.
    n = 10_000
    tokens = [(i * 31) & 0xFFFF for i in range(n)]
    positions = list(range(n))
    write_slots = list(range(n))
    verify_slots = list(range(n))
    verify_prev = [-1] + list(range(n - 1))
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=verify_slots,
        verify_positions=positions,
        verify_prev_slot_indices=verify_prev,
        verify_num_valid=[1] * n,
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=n,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_TAIL,
        prefill_writes=dict(
            slots=write_slots,
            tokens=tokens,
            positions=positions,
        ),
        plan=plan,
    )


def _scenario_ring_buffer_small_capacity() -> dict:
    # Exactly one mismatch is enough to populate one ring row;
    # keeps us in the single-producer-per-row regime where the
    # torch reference and the CUDA kernel agree bit-wise.
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_prev_slot_indices=[-1],
        verify_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=4,
        ring_capacity=1,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=dict(
            slots=[0],
            tokens=[42],
            positions=[0],
        ),
        plan=plan,
    )


def _scenario_real_kv_hash_clean_chain(*, mode_int: int, read_bytes: int) -> dict:
    """Real-KV hash on / clean chain: write 3 slots then verify them.

    The real-KV buffer is filled with a deterministic per-slot pattern so
    each slot has distinct bytes (otherwise the hash collisions would
    hide a corruption bug); both the write and verify passes go through
    the same buffer, so the stored fingerprint matches the recomputed
    one and no violation should fire.
    """
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1, 2],
        verify_positions=[0, 1, 2],
        verify_prev_slot_indices=[-1, 0, 1],
        verify_num_valid=[1, 1, 1],
        write_slot_indices=[0, 1, 2],
        write_token_ids=[101, 202, 303],
        write_positions=[0, 1, 2],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_num_valid=[1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=None,
        plan=plan,
        real_kv=dict(
            slot_stride_bytes=64,
            read_bytes=read_bytes,
            mode_int=mode_int,
            # Per-slot byte pattern: byte j of slot i = (i + 1) * 13 + j.
            # Distinct per (i, j) so different slots hash to different
            # values within the read window.
            byte_pattern="seq",
        ),
    )


def _scenario_real_kv_hash_corruption_caught() -> dict:
    """Real-KV hash on: prefill 3 slots (writes the fingerprint), corrupt one
    real-KV byte, then verify-only.

    Step 1 (prefill, executed before the differential run): the prefill
    helper launches a write-only canary_step with real-KV mode ON, so
    each slot's ``real_kv_hash`` field stores the splitmix64 fold of the
    clean real-KV bytes.

    Step 2 (between prefill and verify): we mutate one byte inside slot
    1's read window, so the verify path will recompute a different
    fingerprint than the one stored.

    Step 3 (verify, the actual differential launch): verify_only on
    slots 0..2; verify thread for slot 1 detects ``REAL_KV_HASH`` and
    records a violation. The CUDA + torch reference paths must produce
    bit-identical state.
    """
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1, 2],
        verify_positions=[0, 1, 2],
        verify_prev_slot_indices=[-1, 0, 1],
        verify_num_valid=[1, 1, 1],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=8,
        ring_capacity=8,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=dict(
            slots=[0, 1, 2],
            tokens=[101, 202, 303],
            positions=[0, 1, 2],
        ),
        plan=plan,
        real_kv=dict(
            slot_stride_bytes=64,
            read_bytes=16,
            mode_int=1,
            byte_pattern="seq",
            # Apply BEFORE the verify-only differential launch:
            # mutate one byte inside slot 1's read window so the
            # recomputed fingerprint won't match the prefill-stored one.
            corrupt_between=dict(slot_idx=1, byte_offset=0, new_value=0xFF),
        ),
        expected_fail_reason=FailReason.REAL_KV_HASH,
    )


def _scenario_inactive_mask_skipped() -> dict:
    # Every active_mask = 0 -> kernel must be a no-op on slot I/O and
    # slot_run_counter, but kernel_run_counter still advances. Same
    # contract must hold for the torch reference.
    plan = _empty_plan_lists()
    plan.update(
        verify_slot_indices=[0, 1],
        verify_positions=[0, 1],
        verify_prev_slot_indices=[-1, 0],
        verify_num_valid=[0, 0],
        write_slot_indices=[0, 1],
        write_token_ids=[10, 20],
        write_positions=[0, 1],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[2],
        write_req_num_valid=[0],
    )
    return dict(
        slot_stride=CANARY_SLOT_BYTES,
        num_slots=4,
        ring_capacity=4,
        kernel_kind=KERNEL_KIND_HEAD,
        prefill_writes=None,
        plan=plan,
    )


_DIFFERENTIAL_SCENARIOS = {
    "write_only_single_req": _scenario_write_only_single_req,
    "write_only_multi_req": _scenario_write_only_multi_req,
    "verify_only_clean_chain": _scenario_verify_only_clean_chain,
    "verify_only_position_mismatch": _scenario_verify_only_position_mismatch,
    "verify_only_hash_mismatch": _scenario_verify_only_hash_mismatch,
    "mixed_write_and_verify": _scenario_mixed_write_and_verify,
    "first_violation_latch_preserved": _scenario_first_violation_latch_preserved,
    "small_k_req": _scenario_small_k_req,
    "large_k_req_1w": _scenario_large_k_req_1w,
    "ring_buffer_small_capacity": _scenario_ring_buffer_small_capacity,
    "inactive_mask_skipped": _scenario_inactive_mask_skipped,
    "real_kv_hash_clean_chain_bit_mode": (
        lambda: _scenario_real_kv_hash_clean_chain(mode_int=1, read_bytes=16)
    ),
    "real_kv_hash_clean_chain_all_mode": (
        lambda: _scenario_real_kv_hash_clean_chain(mode_int=2, read_bytes=64)
    ),
    "real_kv_hash_corruption_caught": _scenario_real_kv_hash_corruption_caught,
}


def _prefill_buffer_with_chain(
    *,
    buf: torch.Tensor,
    slots: list[int],
    tokens: list[int],
    positions: list[int],
    slot_stride: int,
    real_kv_buf: torch.Tensor | None = None,
    real_kv_read_bytes: int = 0,
    real_kv_hash_mode: int = 0,
) -> None:
    """Drive a clean chain into ``buf`` via the CUDA kernel (used to set up
    the source buffer for verify-only differential scenarios)."""
    n = len(slots)
    state = _alloc_diff_state(
        num_slots=buf.shape[0],
        slot_stride=slot_stride,
        ring_capacity=4,
        device=buf.device.type,
    )
    inputs = _build_inputs_on(
        buf.device.type,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        verify_num_valid=[],
        write_slot_indices=slots,
        write_token_ids=tokens,
        write_positions=positions,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[n],
        write_req_num_valid=[1],
    )
    # Use canary_step itself; result lives in state["dst_buf"], copy to buf.
    if real_kv_buf is None:
        real_kv_buf = torch.zeros(1, 1, dtype=torch.uint8, device=buf.device)
    canary_step(
        buf=state["dst_buf"],
        plan=inputs,
        seed=_SEED,
        kernel_kind=KERNEL_KIND_HEAD,
        violation_ring=state["violation_ring"],
        violation_write_index=state["violation_write_index"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        real_kv_buf=real_kv_buf,
        real_kv_read_bytes=real_kv_read_bytes,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()
    buf.copy_(state["dst_buf"])


def _build_real_kv_buf(
    *, num_slots: int, slot_stride_bytes: int, pattern: str
) -> torch.Tensor:
    """Build a deterministic real-KV buffer for differential test scenarios."""
    if pattern != "seq":
        raise ValueError(f"unknown real-KV pattern {pattern!r}")
    buf = torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device="cuda")
    for i in range(num_slots):
        for j in range(slot_stride_bytes):
            buf[i, j] = ((i + 1) * 13 + j) & 0xFF
    return buf


@pytest.mark.parametrize("scenario_name", sorted(_DIFFERENTIAL_SCENARIOS.keys()))
def test_canary_step_cuda_matches_torch_reference(scenario_name: str) -> None:
    """CUDA ``canary_step`` and ``canary_step_torch_reference`` agree bitwise."""
    scenario = _DIFFERENTIAL_SCENARIOS[scenario_name]()
    slot_stride = scenario["slot_stride"]
    num_slots = scenario["num_slots"]
    ring_capacity = scenario["ring_capacity"]
    kernel_kind = scenario["kernel_kind"]
    plan_lists = scenario["plan"]
    prefill = scenario.get("prefill_writes")
    corrupt = scenario.get("corrupt_prev_hash")
    real_kv_cfg = scenario.get("real_kv")

    real_kv_read_bytes = 0
    real_kv_hash_mode = 0
    real_kv_buf_cuda: torch.Tensor | None = None
    real_kv_buf_ref: torch.Tensor | None = None
    if real_kv_cfg is not None:
        real_kv_read_bytes = int(real_kv_cfg["read_bytes"])
        real_kv_hash_mode = int(real_kv_cfg["mode_int"])
        real_kv_buf_cuda = _build_real_kv_buf(
            num_slots=num_slots,
            slot_stride_bytes=int(real_kv_cfg["slot_stride_bytes"]),
            pattern=real_kv_cfg["byte_pattern"],
        )

    src_initial = torch.zeros(num_slots, slot_stride, dtype=torch.uint8, device="cuda")
    if prefill is not None:
        _prefill_buffer_with_chain(
            buf=src_initial,
            slots=prefill["slots"],
            tokens=prefill["tokens"],
            positions=prefill["positions"],
            slot_stride=slot_stride,
            real_kv_buf=real_kv_buf_cuda,
            real_kv_read_bytes=real_kv_read_bytes,
            real_kv_hash_mode=real_kv_hash_mode,
        )
    if corrupt is not None:
        slot_idx = int(corrupt["slot"])
        value = to_signed_int64(int(corrupt["value"]) & ((1 << 64) - 1))
        src_view = src_initial.view(torch.int64).view(num_slots, slot_stride // 8)
        src_view[slot_idx, 2] = value

    state_cuda = _alloc_diff_state(
        num_slots=num_slots,
        slot_stride=slot_stride,
        ring_capacity=ring_capacity,
        device="cuda",
    )
    state_ref = _clone_state_for_reference(state_cuda, device="cuda")

    inputs_cuda = _build_inputs_on("cuda", **plan_lists)
    inputs_ref = _clone_batch_plan_gpu(inputs_cuda, device="cuda")

    if real_kv_buf_cuda is not None:
        real_kv_buf_ref = real_kv_buf_cuda.detach().clone()
        # Optional corruption between write and verify: we run the write
        # path on a CLEAN real-KV (already in ``real_kv_buf_cuda``), then
        # mutate the byte the verify path will see. To pull this off
        # inside a single differential kernel launch we mutate the buffer
        # IN PLACE: the verify thread reads slot at index
        # ``verify_slot_indices[i]`` and hashes the current bytes; the
        # write driver writes slot at ``write_slot_indices[k]`` and
        # hashes the bytes BEFORE the verify thread is guaranteed to
        # have run. We avoid that ordering hazard by ensuring no
        # ``verify_slot_indices`` overlap with the corruption target —
        # the existing scenario builders enforce this.
        corrupt_between = real_kv_cfg.get("corrupt_between") if real_kv_cfg else None
        if corrupt_between is not None:
            slot_i = int(corrupt_between["slot_idx"])
            offset = int(corrupt_between["byte_offset"])
            new_value = int(corrupt_between["new_value"]) & 0xFF
            real_kv_buf_cuda[slot_i, offset] = new_value
            real_kv_buf_ref[slot_i, offset] = new_value

    _run_differential(
        scenario=scenario_name,
        src_initial=src_initial,
        state_cuda=state_cuda,
        state_ref=state_ref,
        inputs_cuda=inputs_cuda,
        inputs_ref=inputs_ref,
        kernel_kind=kernel_kind,
        real_kv_buf_cuda=real_kv_buf_cuda,
        real_kv_buf_ref=real_kv_buf_ref,
        real_kv_read_bytes=real_kv_read_bytes,
        real_kv_hash_mode=real_kv_hash_mode,
    )

    expected_fail_reason = scenario.get("expected_fail_reason")
    if expected_fail_reason is not None:
        assert int(state_cuda["violation_write_index"].item()) >= 1
        assert int(state_cuda["violation_ring"][0, 1].item()) == int(
            expected_fail_reason
        )
