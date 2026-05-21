from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from sglang.jit_kernel.tests.kv_canary._differential import _run_both_plan
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    _allocate_plan_pair,
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
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")

_FUZZ_ITER_PER_SEED = 50


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanFuzzInputs:
    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor
    req_to_token: torch.Tensor
    swa_window_size: int
    full_to_swa_index_mapping: Optional[torch.Tensor]
    verify_capacity: int
    write_req_capacity: int


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
    rtt_kind = rng.choice(["linear", "sparse_permuted", "with_holes"])
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
    fb_rpi_list: list[int] = []
    fb_pfx_list: list[int] = []
    fb_ext_list: list[int] = []
    for r in range(bs):
        if padding_mask[r]:
            fb_rpi_list.append(0)
            fb_pfx_list.append(0)
            fb_ext_list.append(0)
        else:
            fb_rpi_list.append(rng.randint(1, max_reqs - 1))
            fb_pfx_list.append(rng.randint(0, max_seq_len - 1))
            fb_ext_list.append(rng.randint(1, max(1, max_seq_len // 4)))
    fb_rpi = torch.tensor(fb_rpi_list, dtype=torch.int64, device=_DEVICE)
    fb_pfx = torch.tensor(fb_pfx_list, dtype=torch.int64, device=_DEVICE)
    fb_ext = torch.tensor(fb_ext_list, dtype=torch.int64, device=_DEVICE)

    total_verify = 0
    for rpi, pfx in zip(fb_rpi_list, fb_pfx_list):
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

    return PlanFuzzInputs(
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_pfx,
        fb_extend_seq_lens=fb_ext,
        req_to_token=rtt,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )


def _run_one(inputs: PlanFuzzInputs) -> tuple:
    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=inputs.verify_capacity,
        write_req_capacity=inputs.write_req_capacity,
    )
    _run_both_plan(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=inputs.fb_req_pool_indices,
        fb_prefix_lens=inputs.fb_prefix_lens,
        fb_extend_seq_lens=inputs.fb_extend_seq_lens,
        req_to_token=inputs.req_to_token,
        extras=(
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.empty(0, dtype=torch.int64, device=_DEVICE),
            torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        ),
        swa_window_size=inputs.swa_window_size,
        full_to_swa_index_mapping=inputs.full_to_swa_index_mapping,
    )
    PlanInvariants.assert_all(
        verify_plan=triton_v,
        write_plan=triton_w,
        fb_req_pool_indices=inputs.fb_req_pool_indices,
        fb_prefix_lens=inputs.fb_prefix_lens,
        fb_extend_seq_lens=inputs.fb_extend_seq_lens,
        swa_window_size=inputs.swa_window_size,
        extras_slot_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_positions=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_prev_slot_indices=torch.empty(0, dtype=torch.int64, device=_DEVICE),
        extras_count=0,
    )
    return triton_v, triton_w


def _summarize(inputs: PlanFuzzInputs) -> str:
    return (
        f"bs={int(inputs.fb_req_pool_indices.shape[0])} "
        f"swa={inputs.swa_window_size} "
        f"verify_cap={inputs.verify_capacity} write_cap={inputs.write_req_capacity} "
        f"has_lut={inputs.full_to_swa_index_mapping is not None}"
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
