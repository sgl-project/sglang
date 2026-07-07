from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from sglang.jit_kernel.tests.kv_canary._differential import _run_both_plan
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    allocate_plan_pair,
    derive_plan_capacity,
    make_lut,
    make_padding_mask,
    make_req_to_token,
)
from sglang.jit_kernel.tests.kv_canary._fuzz_driver import (
    FUZZ_SEEDS_PR,
    run_fuzz_combo,
)
from sglang.jit_kernel.tests.kv_canary._invariants import PlanInvariants
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")

_FUZZ_ITER_PER_SEED = 50


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanFuzzInputs:
    req_pool_indices: torch.Tensor
    prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    req_to_token: torch.Tensor
    swa_window_size: int
    full_to_swa_index_mapping: Optional[torch.Tensor]
    verify_capacity: int
    write_req_capacity: int
    req_to_verify_expected_tokens: Optional[torch.Tensor]
    kv_token_id_vs_position_offset: int


def _draw_random_plan_inputs(rng: random.Random) -> PlanFuzzInputs:
    bs = rng.randint(1, 16)
    max_seq_len = rng.choice([8, 16, 64, 128, 256])
    swa_enabled = rng.random() < 0.5
    swa_window_size = (
        rng.choice([4, 16, 64, max_seq_len, max(2, max_seq_len // 3)])
        if swa_enabled
        else 0
    )
    swa_window_size = min(swa_window_size, max_seq_len)
    lut_kind = (
        rng.choice(["identity", "shift", "permutation", "with_oob"])
        if swa_enabled
        else None
    )
    rtt_kind = rng.choice(["linear", "sparse_permuted"])
    padding_kind = rng.choice(["none", "trailing", "interleaved"])
    capacity_kind = rng.choice(["loose", "tight_match", "under_by_one"])

    max_reqs = max(bs + 2, 4)
    pool_size = max_reqs * max_seq_len
    rtt = make_req_to_token(
        kind=rtt_kind,
        max_reqs=max_reqs,
        max_seq_len=max_seq_len,
        device=_DEVICE,
        rng=rng,
    )
    padding_mask = make_padding_mask(bs=bs, kind=padding_kind, rng=rng)
    req_pool_indices_list: list[int] = []
    prefix_lens_list: list[int] = []
    extend_seq_lens_list: list[int] = []
    for r in range(bs):
        if padding_mask[r]:
            req_pool_indices_list.append(0)
            prefix_lens_list.append(0)
            extend_seq_lens_list.append(0)
        else:
            req_pool_indices_list.append(rng.randint(1, max_reqs - 1))
            prefix_lens_list.append(rng.randint(0, max_seq_len - 1))
            extend_seq_lens_list.append(rng.randint(1, max(1, max_seq_len // 4)))
    req_pool_indices = torch.tensor(
        req_pool_indices_list, dtype=torch.int64, device=_DEVICE
    )
    prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int64, device=_DEVICE)
    extend_seq_lens = torch.tensor(
        extend_seq_lens_list, dtype=torch.int64, device=_DEVICE
    )

    total_verify = 0
    for rpi, pfx in zip(req_pool_indices_list, prefix_lens_list):
        if rpi == 0:
            continue
        if swa_window_size > 0:
            window_start = max(0, pfx - swa_window_size)
            total_verify += max(0, pfx - window_start)
        else:
            total_verify += pfx

    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind=capacity_kind,
        total_verify=total_verify,
        extras_count=0,
        bs=bs,
    )

    full_to_swa: Optional[torch.Tensor]
    if swa_window_size > 0 and lut_kind is not None:
        full_to_swa = make_lut(
            kind=lut_kind, pool_size=pool_size, device=_DEVICE, rng=rng
        )
    else:
        full_to_swa = None

    expected_pool_present = rng.random() < 0.5
    kv_token_id_vs_position_offset = rng.choice([0, 1])
    expected_pool: Optional[torch.Tensor]
    if expected_pool_present:
        pool_max_context_len = rng.choice(
            [
                max(1, max_seq_len // 4),
                max(1, max_seq_len // 2),
                max_seq_len,
            ]
        )
        expected_pool = torch.randint(
            low=0,
            high=50000,
            size=(max_reqs, pool_max_context_len),
            dtype=torch.int32,
            device=_DEVICE,
        )
    else:
        expected_pool = None

    return PlanFuzzInputs(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=rtt,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
        req_to_verify_expected_tokens=expected_pool,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )


def _run_one(inputs: PlanFuzzInputs) -> tuple:
    triton_v, triton_w, ref_v, ref_w = allocate_plan_pair(
        verify_capacity=inputs.verify_capacity,
        write_req_capacity=inputs.write_req_capacity,
    )
    _run_both_plan(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        req_pool_indices=inputs.req_pool_indices,
        prefix_lens=inputs.prefix_lens,
        extend_seq_lens=inputs.extend_seq_lens,
        req_to_token=inputs.req_to_token,
        extras=(
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        ),
        swa_window_size=inputs.swa_window_size,
        full_to_swa_index_mapping=inputs.full_to_swa_index_mapping,
        req_to_verify_expected_tokens=inputs.req_to_verify_expected_tokens,
        kv_token_id_vs_position_offset=inputs.kv_token_id_vs_position_offset,
    )
    PlanInvariants.assert_all(
        verify_plan=triton_v,
        write_plan=triton_w,
        req_pool_indices=inputs.req_pool_indices,
        prefix_lens=inputs.prefix_lens,
        extend_seq_lens=inputs.extend_seq_lens,
        swa_window_size=inputs.swa_window_size,
        extras_slot_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_positions=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_prev_slot_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_count=0,
    )
    return triton_v, triton_w


def _summarize(inputs: PlanFuzzInputs) -> str:
    return (
        f"bs={int(inputs.req_pool_indices.shape[0])} "
        f"swa={inputs.swa_window_size} "
        f"verify_cap={inputs.verify_capacity} write_cap={inputs.write_req_capacity} "
        f"has_lut={inputs.full_to_swa_index_mapping is not None} "
        f"has_pool={inputs.req_to_verify_expected_tokens is not None} "
        f"offset={inputs.kv_token_id_vs_position_offset}"
    )


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_plan_fuzz_full_combo(seed: int) -> None:
    """Multi-dim plan fuzzer: random LUT/rtt/padding/capacity/swa × N iters, byte-equal."""
    run_fuzz_combo(
        seed,
        draw_fn=_draw_random_plan_inputs,
        run_one_fn=_run_one,
        summarize_fn=_summarize,
        n_iter=_FUZZ_ITER_PER_SEED,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
