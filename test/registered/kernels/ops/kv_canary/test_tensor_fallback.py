"""The XPU kv-canary plan / verify / write path must stay byte-equal to the torch references."""

from __future__ import annotations

import random
from typing import Callable, NamedTuple

import pytest
import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary._tensor_fallback import (
    launch_canary_verify_kernel_tensor_fallback,
    launch_canary_write_kernel_tensor_fallback,
)
from sglang.kernels.ops.kv_canary.consts import to_signed_int64
from sglang.kernels.ops.kv_canary.plan import launch_canary_plan_kernels
from sglang.kernels.ops.kv_canary.plan_ref import (
    launch_canary_plan_kernels_torch_reference,
)
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
)
from sglang.kernels.ops.kv_canary.verify_ref import (
    _compute_real_kv_hash_scalar,
    compute_slot_hash,
    launch_canary_verify_kernel_torch_reference,
    materialize_real_kv_sources,
)
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.kernels.ops.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="the tensor fallback runs on XPU only"
)
_DEVICES = [torch.device("xpu")]
_SEEDS = range(40)
_NUM_SLOTS = 512
# Arbitrary sentinel so untouched plan / ring entries are distinguishable from written ones.
_GARBAGE = -7


class _VerifyCase(NamedTuple):
    buf_i64: torch.Tensor
    slots: list[int]
    prev_slots: list[int]
    positions: list[int]
    tokens: list[int]
    sources: tuple[RealKvSource, ...]
    mode: consts.RealKvHashMode
    capacity: int
    num_valid: int


class _WriteCase(NamedTuple):
    buf_i64: torch.Tensor
    req_lens: list[int]
    seeds: list[int]
    slots: list[int]
    tokens: list[int]
    positions: list[int]
    expected_tokens: list[int]
    expected_positions: list[int]
    sources: tuple[RealKvSource, ...]
    mode: consts.RealKvHashMode


class _RunKnobs(NamedTuple):
    ring_capacity: int
    base_idx: int
    flag: bool
    input_assert: bool


def _int64(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _random_knobs(rng: random.Random) -> _RunKnobs:
    return _RunKnobs(
        ring_capacity=rng.choice([4, 64, 2048]),
        base_idx=rng.choice([0, 0, 3, 70]),
        flag=rng.random() < 0.6,
        input_assert=rng.random() < 0.5,
    )


def _random_sources(
    rng: random.Random, mode: consts.RealKvHashMode, device: torch.device
) -> tuple[RealKvSource, ...]:
    if mode == consts.RealKvHashMode.NONE:
        return ()
    return tuple(
        RealKvSource(
            tensor=torch.randint(
                -1000, 1000, (_NUM_SLOTS // page, page * 32), dtype=torch.int16
            ).to(device),
            page_size=page,
            num_bytes_per_token=64,
            read_bytes=rng.choice([16, 48, 64]),
        )
        for page in rng.sample([1, 2, 4], rng.randint(1, 3))
    )


def _make_context(
    *,
    buf_i64: torch.Tensor,
    kernel_kind: CanaryLaunchTag,
    knobs: _RunKnobs,
    sources: tuple[RealKvSource, ...],
    mode: consts.RealKvHashMode,
    device: torch.device,
) -> VerifyOrWriteContext:
    return VerifyOrWriteContext(
        canary_buf=buf_i64.clone().view(torch.uint8).to(device),
        kernel_kind=kernel_kind,
        violation_ring=torch.full(
            (knobs.ring_capacity, consts.VIOLATION_FIELDS),
            _GARBAGE,
            dtype=torch.int64,
            device=device,
        ),
        violation_write_index=torch.full(
            (1,), knobs.base_idx, dtype=torch.int32, device=device
        ),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        real_kv_sources=sources,
        real_kv_hash_mode=mode,
        enable_chain_position_assert=torch.full(
            (1,), int(knobs.flag), dtype=torch.int32, device=device
        ),
    )


def _context_outputs(context: VerifyOrWriteContext) -> list[torch.Tensor]:
    return [
        context.canary_buf,
        context.violation_ring,
        context.violation_write_index,
        context.slot_run_counter,
        context.kernel_run_counter,
    ]


def _assert_all_equal(expected: list[torch.Tensor], actual: list[torch.Tensor]) -> None:
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor.cpu(), actual_tensor.cpu())


def _alloc_plans(
    *, verify_capacity: int, write_req_capacity: int, device: torch.device
) -> tuple[VerifyPlan, WritePlan]:
    verify_plan = VerifyPlan.allocate(verify_capacity=verify_capacity, device=device)
    write_plan = WritePlan.allocate(
        write_req_capacity=write_req_capacity, device=device
    )
    for tensor in (
        verify_plan.verify_slot_indices,
        verify_plan.verify_expected_tokens,
        verify_plan.verify_expected_positions,
        verify_plan.verify_prev_slot_indices,
        verify_plan.verify_num_valid,
        verify_plan.enable,
        write_plan.write_offsets,
        write_plan.write_seed_slot_indices,
        write_plan.write_num_valid_reqs,
    ):
        tensor.fill_(_GARBAGE)
    return verify_plan, write_plan


def _plan_outputs(verify_plan: VerifyPlan, write_plan: WritePlan) -> list[torch.Tensor]:
    return [
        verify_plan.verify_slot_indices,
        verify_plan.verify_expected_tokens,
        verify_plan.verify_expected_positions,
        verify_plan.verify_prev_slot_indices,
        verify_plan.verify_num_valid,
        verify_plan.enable,
        write_plan.write_offsets,
        write_plan.write_seed_slot_indices,
        write_plan.write_num_valid_reqs,
    ]


@pytest.mark.parametrize("device", _DEVICES, ids=str)
@pytest.mark.parametrize("seed", _SEEDS)
def test_plan_fallback_matches_reference(device: torch.device, seed: int) -> None:
    rng = random.Random(seed)
    num_pool_reqs, max_len, num_kv_slots = 9, 96, 4096
    batch_size = rng.randint(1, 8)
    swa_window_size = rng.choice([0, 0, 7, 40])
    req_to_token = torch.randint(
        0, num_kv_slots, (num_pool_reqs, max_len), dtype=torch.int32
    )
    req_to_token[:, rng.randint(0, max_len - 1) :] = 0
    req_to_token = req_to_token.contiguous()
    full_to_swa_index_mapping = None
    if swa_window_size > 0 or rng.random() < 0.3:
        full_to_swa_index_mapping = torch.randperm(num_kv_slots, dtype=torch.int64).to(
            device
        )
    expected_token_pool = expected_valid_lens = None
    if rng.random() < 0.5:
        expected_token_pool = torch.randint(
            0, 1000, (num_pool_reqs, max_len + 4), dtype=torch.int64, device=device
        )
        expected_valid_lens = _int64(
            [rng.randint(0, max_len) for _ in range(batch_size)], device
        )
    verify_capacity = rng.choice([16, 256, 4096])
    write_req_capacity = rng.choice([batch_size, batch_size + 3])
    inputs = dict(
        req_pool_indices=_int64(
            [rng.randrange(num_pool_reqs) for _ in range(batch_size)], device
        ),
        prefix_lens=_int64(
            [rng.randint(0, max_len - 8) for _ in range(batch_size)], device
        ),
        extend_seq_lens=_int64(
            [rng.choice([0, 1, rng.randint(1, 8)]) for _ in range(batch_size)], device
        ),
        req_to_token=req_to_token.to(device),
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=verify_capacity,
        req_to_verify_expected_tokens=expected_token_pool,
        req_to_verify_expected_tokens_valid_lens=expected_valid_lens,
        kv_token_id_vs_position_offset=rng.choice([0, 1]),
    )

    outputs = []
    for launch in (
        launch_canary_plan_kernels_torch_reference,
        launch_canary_plan_kernels,
    ):
        verify_plan, write_plan = _alloc_plans(
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
            device=device,
        )
        launch(verify_plan_out=verify_plan, write_plan_out=write_plan, **inputs)
        outputs.append(_plan_outputs(verify_plan, write_plan))
    _assert_all_equal(*outputs)


def _make_verify_case(rng: random.Random, device: torch.device) -> _VerifyCase:
    buf_i64 = torch.randint(-(2**62), 2**62, (_NUM_SLOTS, 4), dtype=torch.int64)
    # Make about half the slots one valid chain so both the pass and the fail paths are hit.
    chain_slots = list(range(1, _NUM_SLOTS))
    rng.shuffle(chain_slots)
    chain_slots = chain_slots[: _NUM_SLOTS // 2]
    true_prev: dict[int, int] = {}
    prev_slot = -1
    for position, slot in enumerate(chain_slots):
        true_prev[slot] = prev_slot
        buf_i64[slot, consts.CANARY_FIELD_TOKEN] = rng.randint(0, 50000)
        buf_i64[slot, consts.CANARY_FIELD_POSITION] = position
        buf_i64[slot, consts.CANARY_FIELD_PREV_HASH] = to_signed_int64(
            compute_slot_hash(buf_i64, prev_slot)
        )
        prev_slot = slot
    slots = chain_slots + rng.sample(range(_NUM_SLOTS), 40)
    rng.shuffle(slots)

    mode = rng.choice(list(consts.RealKvHashMode))
    sources = _random_sources(rng, mode, device)
    if sources:
        host_sources = materialize_real_kv_sources(
            real_kv_sources=sources,
            real_kv_hash_mode=mode,
            slot_indices=slots,
            work_device=torch.device("cpu"),
        )
        for slot in slots:
            if rng.random() < 0.8:
                buf_i64[slot, consts.CANARY_FIELD_REAL_KV_HASH] = to_signed_int64(
                    _compute_real_kv_hash_scalar(
                        slot_idx=slot, host_sources=host_sources
                    )
                )
    else:
        keep_zero = torch.rand(_NUM_SLOTS) < 0.8
        buf_i64[keep_zero, consts.CANARY_FIELD_REAL_KV_HASH] = 0

    return _VerifyCase(
        buf_i64=buf_i64,
        slots=slots,
        prev_slots=[
            (
                true_prev[slot]
                if slot in true_prev and rng.random() < 0.8
                else rng.choice([-1, 0, rng.randrange(_NUM_SLOTS)])
            )
            for slot in slots
        ],
        positions=[
            int(buf_i64[slot, consts.CANARY_FIELD_POSITION])
            if rng.random() < 0.8
            else 3
            for slot in slots
        ],
        tokens=[
            (
                int(buf_i64[slot, consts.CANARY_FIELD_TOKEN])
                if rng.random() < 0.8
                else rng.choice([-1, 5])
            )
            for slot in slots
        ],
        sources=sources,
        mode=mode,
        capacity=len(slots) + rng.choice([0, 5]),
        num_valid=rng.choice([len(slots), len(slots) // 2]),
    )


def _run_verify(
    launch: Callable[..., None],
    case: _VerifyCase,
    knobs: _RunKnobs,
    device: torch.device,
) -> list[torch.Tensor]:
    num_entries = len(case.slots)
    plan = VerifyPlan.allocate(verify_capacity=case.capacity, device=device)
    plan.verify_slot_indices.fill_(0)
    plan.verify_slot_indices[:num_entries] = _int64(case.slots, device)
    plan.verify_prev_slot_indices[:num_entries] = _int64(case.prev_slots, device)
    plan.verify_expected_positions[:num_entries] = _int64(case.positions, device)
    plan.verify_expected_tokens[:num_entries] = _int64(case.tokens, device)
    plan.verify_num_valid.fill_(case.num_valid)
    # knobs.flag doubles as plan.enable here; verify is off for about 40% of cases.
    plan.enable.fill_(int(knobs.flag))
    context = _make_context(
        buf_i64=case.buf_i64,
        kernel_kind=list(CanaryLaunchTag)[-1],
        knobs=knobs,
        sources=case.sources,
        mode=case.mode,
        device=device,
    )
    launch(context=context, plan=plan, check_verify_expected_token=knobs.input_assert)
    return _context_outputs(context)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
@pytest.mark.parametrize("seed", _SEEDS)
def test_verify_fallback_matches_reference(device: torch.device, seed: int) -> None:
    rng = random.Random(seed)
    case = _make_verify_case(rng, device)
    knobs = _random_knobs(rng)
    _assert_all_equal(
        _run_verify(launch_canary_verify_kernel_torch_reference, case, knobs, device),
        _run_verify(launch_canary_verify_kernel_tensor_fallback, case, knobs, device),
    )


def _make_write_case(rng: random.Random, device: torch.device) -> _WriteCase:
    buf_i64 = torch.randint(-(2**62), 2**62, (_NUM_SLOTS, 4), dtype=torch.int64)
    req_lens = [
        rng.choice([0, 1, 1, rng.randint(2, 30)]) for _ in range(rng.randint(1, 6))
    ]
    total = sum(req_lens)
    # Slots are mostly distinct; a few cases reuse one (see below).
    distinct_slots = rng.sample(range(1, _NUM_SLOTS), total + len(req_lens))
    slots = [slot if rng.random() < 0.9 else -1 for slot in distinct_slots[:total]]
    seeds = [slot if rng.random() < 0.7 else -1 for slot in distinct_slots[total:]]
    written = [slot for slot in slots if slot >= 0]
    if written and rng.random() < 0.3:
        # A later request seeds from a slot an earlier request writes in this same launch.
        seeds[-1] = written[0]
    if len(written) > 1 and rng.random() < 0.2:
        # The same slot written twice in one launch: the last write wins.
        slots[slots.index(written[-1])] = written[0]
    positions: list[int] = []
    for seed_slot, req_len in zip(seeds, req_lens):
        base = rng.randint(0, 100)
        if seed_slot >= 0:
            buf_i64[seed_slot, consts.CANARY_FIELD_POSITION] = base
        positions += [
            base + 1 + offset if rng.random() < 0.9 else rng.randint(0, 9)
            for offset in range(req_len)
        ]
    tokens = [rng.randint(0, 50000) for _ in range(total)]
    mode = rng.choice(list(consts.RealKvHashMode))
    return _WriteCase(
        buf_i64=buf_i64,
        req_lens=req_lens,
        seeds=seeds,
        slots=slots,
        tokens=tokens,
        positions=positions,
        expected_tokens=[
            token if rng.random() < 0.8 else token + 1 for token in tokens
        ],
        expected_positions=[
            position if rng.random() < 0.8 else position + 1 for position in positions
        ],
        sources=_random_sources(rng, mode, device),
        mode=mode,
    )


def _run_write(
    launch: Callable[..., None],
    case: _WriteCase,
    knobs: _RunKnobs,
    device: torch.device,
) -> list[torch.Tensor]:
    num_reqs = len(case.req_lens)
    plan = WritePlan.allocate(write_req_capacity=num_reqs + 2, device=device)
    plan.write_offsets.zero_()
    plan.write_offsets[1 : num_reqs + 1] = _int64(case.req_lens, device).cumsum(0)
    plan.write_seed_slot_indices.fill_(-1)
    plan.write_seed_slot_indices[:num_reqs] = _int64(case.seeds, device)
    plan.write_num_valid_reqs.fill_(num_reqs)
    context = _make_context(
        buf_i64=case.buf_i64,
        kernel_kind=list(CanaryLaunchTag)[0],
        knobs=knobs,
        sources=case.sources,
        mode=case.mode,
        device=device,
    )
    launch(
        context=context,
        plan=plan,
        input_ids=_int64(case.tokens, device),
        positions=_int64(case.positions, device),
        out_cache_loc=_int64(case.slots, device),
        enable_write_input_assert=knobs.input_assert,
        expected_input_tokens=(
            _int64(case.expected_tokens, device) if knobs.input_assert else None
        ),
        expected_input_positions=(
            _int64(case.expected_positions, device) if knobs.input_assert else None
        ),
    )
    return _context_outputs(context)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
@pytest.mark.parametrize("seed", _SEEDS)
def test_write_fallback_matches_reference(device: torch.device, seed: int) -> None:
    rng = random.Random(seed)
    case = _make_write_case(rng, device)
    knobs = _random_knobs(rng)
    _assert_all_equal(
        _run_write(launch_canary_write_kernel_torch_reference, case, knobs, device),
        _run_write(launch_canary_write_kernel_tensor_fallback, case, knobs, device),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
