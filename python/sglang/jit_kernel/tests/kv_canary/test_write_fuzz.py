"""Random differential fuzz tests: CUDA canary_write_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode, WritePlan
from sglang.jit_kernel.tests.kv_canary._differential import (
    ShrinkResult,
    _run_both_write,
    shrink_inputs,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    clone_real_kv_sources,
    make_real_kv_sources,
)
from sglang.jit_kernel.tests.kv_canary._invariants import assert_all_write_invariants
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_write_plan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")

_FUZZ_SEEDS_PR = [0]
_FUZZ_SEEDS_NIGHTLY = list(range(10))
_FUZZ_ITER_PER_SEED = 30


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteFuzzInputs:
    cuda_canary_buf: torch.Tensor
    ref_canary_buf: torch.Tensor
    plan_cuda: WritePlan
    plan_ref: WritePlan
    fb_input_ids: torch.Tensor
    fb_positions: torch.Tensor
    fb_out_cache_loc: torch.Tensor
    kernel_kind: CanaryLaunchTag
    pseudo_mode: CanaryPseudoMode
    pseudo_expected_tokens: torch.Tensor
    pseudo_expected_positions: torch.Tensor
    real_kv_sources_cuda: tuple[RealKvSource, ...]
    real_kv_sources_ref: tuple[RealKvSource, ...]
    real_kv_hash_mode: RealKvHashMode
    ring_capacity: int


def _draw_random_write_inputs(rng: random.Random) -> WriteFuzzInputs:
    pseudo_mode = rng.choice([CanaryPseudoMode.OFF, CanaryPseudoMode.ON])
    hash_mode = rng.choice(
        [RealKvHashMode.OFF, RealKvHashMode.PARTIAL, RealKvHashMode.ALL]
    )
    src_count = rng.choice([1, 2, 4])
    page_size = rng.choice([1, 16])
    bytes_per = rng.choice([8, 64, 128])
    kernel_kind = rng.choice(list(CanaryLaunchTag))
    ring_capacity = rng.choice([16, 64, 256])

    n_reqs = rng.randint(1, 4)
    per_req_tokens: list[int] = [rng.randint(1, 5) for _ in range(n_reqs)]
    total_tokens = sum(per_req_tokens)
    num_slots = max(total_tokens + 8, 16)

    sources_cuda = make_real_kv_sources(
        count=src_count,
        num_bytes_per_token=bytes_per,
        page_size=page_size,
        num_slots=num_slots,
        device=_DEVICE,
        rng=rng,
    )
    sources_ref = clone_real_kv_sources(sources_cuda)

    cuda_buf = make_canary_buf(
        num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
    )
    ref_buf = cuda_buf.clone()

    write_offsets: list[int] = [0]
    running = 0
    for t in per_req_tokens:
        running += t
        write_offsets.append(running)
    slot_pool = list(range(1, num_slots))
    rng.shuffle(slot_pool)
    seed_slot_indices: list[int] = []
    for _ in range(n_reqs):
        if rng.random() < 0.4 or len(slot_pool) <= total_tokens:
            seed_slot_indices.append(-1)
        else:
            seed_slot_indices.append(slot_pool.pop())

    out_cache_loc_list: list[int] = []
    for _ in range(total_tokens):
        if rng.random() < 0.15 and total_tokens > 1:
            out_cache_loc_list.append(-1)
        else:
            if not slot_pool:
                out_cache_loc_list.append(-1)
            else:
                out_cache_loc_list.append(slot_pool.pop())

    plan_cuda = make_write_plan(
        write_offsets=write_offsets,
        seed_slot_indices=seed_slot_indices,
        num_valid_reqs=n_reqs,
        device=_DEVICE,
    )
    plan_ref = make_write_plan(
        write_offsets=write_offsets,
        seed_slot_indices=seed_slot_indices,
        num_valid_reqs=n_reqs,
        device=_DEVICE,
    )

    fb_input_ids = torch.tensor(
        [rng.randint(0, 0xFFFFFFFF) for _ in range(total_tokens)],
        dtype=torch.int32,
        device=_DEVICE,
    )
    fb_positions = torch.tensor(
        [rng.randint(0, 1024) for _ in range(total_tokens)],
        dtype=torch.int32,
        device=_DEVICE,
    )
    fb_out_cache_loc = torch.tensor(
        out_cache_loc_list, dtype=torch.int32, device=_DEVICE
    )
    pseudo_expected_tokens = fb_input_ids.clone()
    pseudo_expected_positions = fb_positions.clone()

    return WriteFuzzInputs(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=hash_mode,
        ring_capacity=ring_capacity,
    )


def _run_one(inputs: WriteFuzzInputs) -> None:
    cuda_buf_before = inputs.cuda_canary_buf.clone()
    cuda_log = FakeViolationLog.allocate(capacity=inputs.ring_capacity, device=_DEVICE)
    ref_log = FakeViolationLog.allocate(capacity=inputs.ring_capacity, device=_DEVICE)
    log_before = FakeViolationLog.allocate(
        capacity=inputs.ring_capacity, device=_DEVICE
    )
    _run_both_write(
        cuda_canary_buf=inputs.cuda_canary_buf,
        ref_canary_buf=inputs.ref_canary_buf,
        plan_cuda=inputs.plan_cuda,
        plan_ref=inputs.plan_ref,
        fb_input_ids=inputs.fb_input_ids,
        fb_positions=inputs.fb_positions,
        fb_out_cache_loc=inputs.fb_out_cache_loc,
        pseudo_mode=inputs.pseudo_mode,
        pseudo_expected_tokens=inputs.pseudo_expected_tokens,
        pseudo_expected_positions=inputs.pseudo_expected_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=inputs.real_kv_sources_cuda,
        real_kv_sources_ref=inputs.real_kv_sources_ref,
        real_kv_hash_mode=inputs.real_kv_hash_mode,
        kernel_kind=inputs.kernel_kind,
    )
    assert torch.equal(
        inputs.cuda_canary_buf, inputs.ref_canary_buf
    ), "CUDA vs ref canary_buf diverged"
    assert int(cuda_log.write_index[0].item()) == int(ref_log.write_index[0].item())
    assert int(cuda_log.slot_run_counter[0].item()) == int(
        ref_log.slot_run_counter[0].item()
    )
    assert int(cuda_log.kernel_run_counter[0].item()) == int(
        ref_log.kernel_run_counter[0].item()
    )
    assert_all_write_invariants(
        canary_buf_before=cuda_buf_before,
        canary_buf_after=inputs.cuda_canary_buf,
        plan=inputs.plan_cuda,
        fb_input_ids=inputs.fb_input_ids,
        fb_positions=inputs.fb_positions,
        fb_out_cache_loc=inputs.fb_out_cache_loc,
        pseudo_mode=inputs.pseudo_mode,
        pseudo_expected_tokens=inputs.pseudo_expected_tokens,
        pseudo_expected_positions=inputs.pseudo_expected_positions,
        log_before=log_before,
        log_after=cuda_log,
    )


def _check_repro(inputs: WriteFuzzInputs) -> bool:
    try:
        _run_one(inputs)
    except (AssertionError, RuntimeError, ValueError):
        return True
    return False


def _summarize(inputs: WriteFuzzInputs) -> str:
    n_active = int(inputs.plan_cuda.write_num_valid_reqs[0].item())
    total = int(inputs.plan_cuda.write_offsets[n_active].item())
    return (
        f"n_reqs={n_active} total_tokens={total} kind={inputs.kernel_kind.name} "
        f"pseudo={inputs.pseudo_mode.name} hash_mode={inputs.real_kv_hash_mode.name} "
        f"sources={len(inputs.real_kv_sources_cuda)}"
    )


@pytest.mark.parametrize("seed", _FUZZ_SEEDS_PR)
def test_write_fuzz_full_combo(seed: int) -> None:
    """Multi-dim write fuzzer: random pseudo/hash/kernel/page/source × N iters, byte-equal."""
    rng = random.Random(seed)
    for iteration in range(_FUZZ_ITER_PER_SEED):
        inputs = _draw_random_write_inputs(rng)
        try:
            _run_one(inputs)
        except AssertionError as exc:
            shrunk: ShrinkResult = shrink_inputs(inputs, check_fn=_check_repro)
            raise AssertionError(
                f"seed={seed} iter={iteration} failure: {exc}\n"
                f"original: {_summarize(inputs)}\n"
                f"shrunk:   {_summarize(shrunk.inputs)}\n"
                f"mutations applied: {shrunk.mutations_applied}"
            ) from exc
