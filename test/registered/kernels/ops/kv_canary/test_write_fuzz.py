from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
)
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_log_pair,
    make_write_plan_pair,
    stamp_pair,
)
from sglang.test.kernels.kv_canary._differential import _run_both_write
from sglang.test.kernels.kv_canary._fixtures import (
    clone_real_kv_sources,
    make_real_kv_sources,
)
from sglang.test.kernels.kv_canary._fuzz_driver import (
    FUZZ_SEEDS_PR,
    run_fuzz_combo,
)
from sglang.test.kernels.kv_canary._invariants import WriteInvariants

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")

_FUZZ_ITER_PER_SEED = 30


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteFuzzInputs:
    cuda_canary_buf: torch.Tensor
    ref_canary_buf: torch.Tensor
    plan_cuda: WritePlan
    plan_ref: WritePlan
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    kernel_kind: CanaryLaunchTag
    enable_write_verify_inputs: bool
    expected_input_tokens: torch.Tensor
    expected_input_positions: torch.Tensor
    real_kv_sources_cuda: tuple[RealKvSource, ...]
    real_kv_sources_ref: tuple[RealKvSource, ...]
    real_kv_hash_mode: consts.RealKvHashMode
    ring_capacity: int


def _draw_random_write_inputs(rng: random.Random) -> WriteFuzzInputs:
    enable_write_verify_inputs = rng.choice([False, True])
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
        if not slot_pool:
            out_cache_loc_list.append(-1)
        else:
            out_cache_loc_list.append(slot_pool.pop())

    plan_cuda, plan_ref = make_write_plan_pair(
        write_offsets=write_offsets,
        seed_slot_indices=seed_slot_indices,
        num_valid_reqs=n_reqs,
        device=_DEVICE,
    )

    input_ids = torch.tensor(
        [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(total_tokens)],
        dtype=torch.int64,
        device=_DEVICE,
    )
    # Per-chain sequential positions so the write kernel's chain-step position assert holds.
    # For chains with a real seed slot, stamp the seed with (chain_start_position - 1) so the
    # first chain entry's position == seed.position + 1.
    chain_start_positions: list[int] = [rng.randint(0, 1024) for _ in range(n_reqs)]
    positions_list: list[int] = []
    for r in range(n_reqs):
        start = chain_start_positions[r]
        positions_list.extend(start + i for i in range(per_req_tokens[r]))
        seed_slot = seed_slot_indices[r]
        if seed_slot >= 0:
            stamp_pair(
                (cuda_buf, ref_buf),
                slot_idx=seed_slot,
                token=0,
                position=start - 1,
                prev_hash=0,
            )
    positions = torch.tensor(positions_list, dtype=torch.int64, device=_DEVICE)
    out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int64, device=_DEVICE)
    expected_input_tokens = input_ids.clone()
    expected_input_positions = positions.clone()
    if enable_write_verify_inputs:
        candidate_indices = [
            idx for idx, slot in enumerate(out_cache_loc_list) if slot >= 0
        ]
        rng.shuffle(candidate_indices)
        mismatch_count = rng.randint(0, len(candidate_indices))
        for idx in candidate_indices[:mismatch_count]:
            if rng.choice([False, True]):
                expected_input_tokens[idx] = expected_input_tokens[idx] + 1
            else:
                expected_input_positions[idx] = expected_input_positions[idx] + 1

    return WriteFuzzInputs(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        kernel_kind=kernel_kind,
        enable_write_verify_inputs=enable_write_verify_inputs,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=hash_mode,
        ring_capacity=ring_capacity,
    )


def _run_one(inputs: WriteFuzzInputs) -> None:
    cuda_buf_before = inputs.cuda_canary_buf.clone()
    cuda_log, ref_log = make_log_pair(capacity=inputs.ring_capacity, device=_DEVICE)
    log_before = FakeViolationLog.allocate(
        capacity=inputs.ring_capacity, device=_DEVICE
    )
    _run_both_write(
        cuda_canary_buf=inputs.cuda_canary_buf,
        ref_canary_buf=inputs.ref_canary_buf,
        plan_cuda=inputs.plan_cuda,
        plan_ref=inputs.plan_ref,
        input_ids=inputs.input_ids,
        positions=inputs.positions,
        out_cache_loc=inputs.out_cache_loc,
        enable_write_verify_inputs=inputs.enable_write_verify_inputs,
        expected_input_tokens=inputs.expected_input_tokens,
        expected_input_positions=inputs.expected_input_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=inputs.real_kv_sources_cuda,
        real_kv_sources_ref=inputs.real_kv_sources_ref,
        real_kv_hash_mode=inputs.real_kv_hash_mode,
        kernel_kind=inputs.kernel_kind,
        assert_equal=False,
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
    WriteInvariants.assert_all(
        canary_buf_before=cuda_buf_before,
        canary_buf_after=inputs.cuda_canary_buf,
        plan=inputs.plan_cuda,
        input_ids=inputs.input_ids,
        positions=inputs.positions,
        out_cache_loc=inputs.out_cache_loc,
        enable_write_verify_inputs=inputs.enable_write_verify_inputs,
        expected_input_tokens=inputs.expected_input_tokens,
        expected_input_positions=inputs.expected_input_positions,
        log_before=log_before,
        log_after=cuda_log,
    )


def _summarize(inputs: WriteFuzzInputs) -> str:
    n_active = int(inputs.plan_cuda.write_num_valid_reqs[0].item())
    total = int(inputs.plan_cuda.write_offsets[n_active].item())
    return (
        f"n_reqs={n_active} total_tokens={total} kind={inputs.kernel_kind.name} "
        f"pseudo={inputs.enable_write_verify_inputs} hash_mode={inputs.real_kv_hash_mode.name} "
        f"sources={len(inputs.real_kv_sources_cuda)}"
    )


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_write_fuzz_full_combo(seed: int) -> None:
    """Multi-dim write fuzzer: random pseudo/hash/kernel/page/source × N iters, byte-equal."""
    run_fuzz_combo(
        seed,
        draw_fn=_draw_random_write_inputs,
        run_one_fn=_run_one,
        summarize_fn=_summarize,
        n_iter=_FUZZ_ITER_PER_SEED,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
