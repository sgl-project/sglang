"""Random differential fuzz tests: CUDA canary_verify_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_CHAIN_ANCHOR,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
    VerifyPlan,
)
from sglang.jit_kernel.tests.kv_canary._differential import (
    ShrinkResult,
    _run_both_verify,
    shrink_inputs,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    clone_real_kv_sources,
    make_real_kv_sources,
)
from sglang.jit_kernel.tests.kv_canary._invariants import assert_all_verify_invariants
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_verify_plan,
    splitmix64,
    splitmix64_mix4,
    to_signed_int64,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")

_FUZZ_SEEDS_PR = [0]
_FUZZ_SEEDS_NIGHTLY = list(range(10))
_FUZZ_ITER_PER_SEED = 30


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyFuzzInputs:
    cuda_canary_buf: torch.Tensor
    ref_canary_buf: torch.Tensor
    plan_cuda: VerifyPlan
    plan_ref: VerifyPlan
    kernel_kind: CanaryLaunchTag
    real_kv_sources_cuda: tuple[RealKvSource, ...]
    real_kv_sources_ref: tuple[RealKvSource, ...]
    real_kv_hash_mode: RealKvHashMode
    ring_capacity: int


def _stamp_clean_chain(
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    slot_indices: list[int],
    tokens: list[int],
    positions: list[int],
) -> None:
    running = splitmix64(CANARY_CHAIN_ANCHOR)
    for slot_idx, token, position in zip(slot_indices, tokens, positions):
        signed_prev = to_signed_int64(running)
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=signed_prev,
                real_kv_hash=0,
            )
        running = splitmix64_mix4(running, token, position, 0)


def _draw_random_verify_inputs(rng: random.Random) -> VerifyFuzzInputs:
    hash_mode = rng.choice([RealKvHashMode.OFF, RealKvHashMode.BIT, RealKvHashMode.ALL])
    src_count = rng.choice([1, 2, 4])
    page_size = rng.choice([1, 16])
    bytes_per = rng.choice([8, 64, 128])
    kernel_kind = rng.choice(list(CanaryLaunchTag))
    plan_size = rng.randint(0, 32)
    num_slots = max(plan_size + 8, 16)
    ring_capacity = rng.choice([16, 64, 256])

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

    slot_universe = list(range(1, num_slots))
    rng.shuffle(slot_universe)
    slot_indices = slot_universe[:plan_size]
    tokens = [rng.randint(0, 0xFFFFFFFF) for _ in range(plan_size)]
    positions = list(range(plan_size))
    prev_slot_indices: list[int] = []
    for i in range(plan_size):
        if i == 0:
            prev_slot_indices.append(-1)
        else:
            prev_slot_indices.append(slot_indices[i - 1])

    if hash_mode == RealKvHashMode.OFF and plan_size > 0:
        _stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            slot_indices=slot_indices,
            tokens=tokens,
            positions=positions,
        )

    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        capacity=max(plan_size, 1),
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        capacity=max(plan_size, 1),
        device=_DEVICE,
    )

    return VerifyFuzzInputs(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        kernel_kind=kernel_kind,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=hash_mode,
        ring_capacity=ring_capacity,
    )


def _run_one(inputs: VerifyFuzzInputs) -> None:
    cuda_buf_before = inputs.cuda_canary_buf.clone()
    cuda_log = FakeViolationLog.allocate(capacity=inputs.ring_capacity, device=_DEVICE)
    ref_log = FakeViolationLog.allocate(capacity=inputs.ring_capacity, device=_DEVICE)
    log_before = FakeViolationLog.allocate(
        capacity=inputs.ring_capacity, device=_DEVICE
    )
    _run_both_verify(
        cuda_canary_buf=inputs.cuda_canary_buf,
        ref_canary_buf=inputs.ref_canary_buf,
        plan_cuda=inputs.plan_cuda,
        plan_ref=inputs.plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=inputs.real_kv_sources_cuda,
        real_kv_sources_ref=inputs.real_kv_sources_ref,
        real_kv_hash_mode=inputs.real_kv_hash_mode,
        kernel_kind=inputs.kernel_kind,
    )
    assert int(cuda_log.kernel_run_counter[0].item()) == int(
        ref_log.kernel_run_counter[0].item()
    )
    assert int(cuda_log.slot_run_counter[0].item()) == int(
        ref_log.slot_run_counter[0].item()
    )
    assert int(cuda_log.write_index[0].item()) == int(ref_log.write_index[0].item())
    assert_all_verify_invariants(
        canary_buf_before=cuda_buf_before,
        canary_buf_after=inputs.cuda_canary_buf,
        log_before=log_before,
        log_after=cuda_log,
        plan=inputs.plan_cuda,
        kernel_kind=inputs.kernel_kind,
    )


def _check_repro(inputs: VerifyFuzzInputs) -> bool:
    try:
        _run_one(inputs)
    except (AssertionError, RuntimeError, ValueError):
        return True
    return False


def _summarize(inputs: VerifyFuzzInputs) -> str:
    n_active = int(inputs.plan_cuda.verify_num_valid[0].item())
    return (
        f"plan_size={n_active} kind={inputs.kernel_kind.name} "
        f"hash_mode={inputs.real_kv_hash_mode.name} "
        f"sources={len(inputs.real_kv_sources_cuda)} "
        f"ring={inputs.ring_capacity}"
    )


@pytest.mark.parametrize("seed", _FUZZ_SEEDS_PR)
def test_verify_fuzz_full_combo(seed: int) -> None:
    """Multi-dim verify fuzzer: random hash mode × kernel kind × page × bytes × N iters, byte-equal."""
    rng = random.Random(seed)
    for iteration in range(_FUZZ_ITER_PER_SEED):
        inputs = _draw_random_verify_inputs(rng)
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
