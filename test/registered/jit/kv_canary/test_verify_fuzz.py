from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from sglang.jit_kernel.tests.kv_canary._canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_log_pair,
    make_verify_plan_pair,
    stamp_clean_chain,
)
from sglang.jit_kernel.tests.kv_canary._differential import _run_both_verify
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    clone_real_kv_sources,
    make_real_kv_sources,
)
from sglang.jit_kernel.tests.kv_canary._fuzz_driver import (
    FUZZ_SEEDS_PR,
    run_fuzz_combo,
)
from sglang.jit_kernel.tests.kv_canary._invariants import VerifyInvariants
from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyPlan,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")

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
    real_kv_hash_mode: consts.RealKvHashMode
    ring_capacity: int
    check_verify_expected_token: bool


def _draw_random_verify_inputs(rng: random.Random) -> VerifyFuzzInputs:
    hash_mode = rng.choice(
        [
            consts.RealKvHashMode.NONE,
            consts.RealKvHashMode.PARTIAL,
            consts.RealKvHashMode.ALL,
        ]
    )
    src_count = rng.choice([1, 2, 4])
    page_size = rng.choice([1, 16])
    bytes_per = rng.choice([16, 64, 128])
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

    if hash_mode == consts.RealKvHashMode.NONE and plan_size > 0:
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            slot_indices=slot_indices,
            tokens=tokens,
            positions=positions,
        )

    # Inject prev_slot == TOKEN_TO_KV_SLOT_PADDING into ~15% of entries so the differential
    # harness exercises the chain-check-skip branch (added for SWA-evicted ancestor handling).
    # Done AFTER stamp_clean_chain so the stored prev_hash on those slots is still chain-clean;
    # the kernel must rely on prev_slot==padding (not on stored hash) to decide whether to skip.
    for i in range(plan_size):
        if rng.random() < 0.15:
            prev_slot_indices[i] = consts.TOKEN_TO_KV_SLOT_PADDING

    check_verify_expected_token = rng.random() < 0.5
    expected_input_ids: list[int] = []
    for i in range(plan_size):
        # Always pick a value; with check=False the kernel must not deref this column.
        if rng.random() < 0.3:
            expected_input_ids.append(-1)
        elif rng.random() < 0.5:
            expected_input_ids.append(int(tokens[i]))
        else:
            mutated = (int(tokens[i]) ^ 0x1) & 0xFFFFFFFF
            expected_input_ids.append(mutated)

    plan_cuda, plan_ref = make_verify_plan_pair(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        expected_input_ids=expected_input_ids if plan_size > 0 else None,
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
        check_verify_expected_token=check_verify_expected_token,
    )


def _run_one(inputs: VerifyFuzzInputs) -> None:
    cuda_buf_before = inputs.cuda_canary_buf.clone()
    cuda_log, ref_log = make_log_pair(capacity=inputs.ring_capacity, device=_DEVICE)
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
        assert_equal=False,
        check_verify_expected_token=inputs.check_verify_expected_token,
    )
    assert int(cuda_log.kernel_run_counter[0].item()) == int(
        ref_log.kernel_run_counter[0].item()
    )
    assert int(cuda_log.slot_run_counter[0].item()) == int(
        ref_log.slot_run_counter[0].item()
    )
    assert int(cuda_log.write_index[0].item()) == int(ref_log.write_index[0].item())
    VerifyInvariants.assert_all(
        canary_buf_before=cuda_buf_before,
        canary_buf_after=inputs.cuda_canary_buf,
        log_before=log_before,
        log_after=cuda_log,
        plan=inputs.plan_cuda,
        kernel_kind=inputs.kernel_kind,
    )


def _summarize(inputs: VerifyFuzzInputs) -> str:
    n_active = int(inputs.plan_cuda.verify_num_valid[0].item())
    return (
        f"plan_size={n_active} kind={inputs.kernel_kind.name} "
        f"hash_mode={inputs.real_kv_hash_mode.name} "
        f"sources={len(inputs.real_kv_sources_cuda)} "
        f"ring={inputs.ring_capacity} "
        f"check_token={inputs.check_verify_expected_token}"
    )


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_verify_fuzz_full_combo(seed: int) -> None:
    """Multi-dim verify fuzzer: random hash mode × kernel kind × page × bytes × N iters, byte-equal."""
    run_fuzz_combo(
        seed,
        draw_fn=_draw_random_verify_inputs,
        run_one_fn=_run_one,
        summarize_fn=_summarize,
        n_iter=_FUZZ_ITER_PER_SEED,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
