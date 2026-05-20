"""Random differential fuzz tests: Triton canary_plan_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from sglang.jit_kernel.tests.kv_canary._differential import (
    ShrinkResult,
    _run_both_and_assert_plan_byte_equal,
    shrink_inputs,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    _allocate_plan_pair,
    derive_plan_capacity,
    make_lut,
    make_padding_mask,
    make_req_to_token,
)
from sglang.jit_kernel.tests.kv_canary._invariants import assert_all_plan_invariants
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")

_FUZZ_SEEDS_PR = [0]
_FUZZ_SEEDS_NIGHTLY = list(range(10))
_FUZZ_ITER_PER_SEED = 50


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanFuzzInputs:
    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor
    req_to_token: torch.Tensor
    extras_slot_indices: torch.Tensor
    extras_positions: torch.Tensor
    extras_prev_slot_indices: torch.Tensor
    extras_num_valid: torch.Tensor
    extras_count: int
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
    extras_kind = rng.choice(["none", "few", "tile_boundary_64", "many_129"])
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
    fb_rpi = torch.tensor(fb_rpi_list, dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor(fb_pfx_list, dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor(fb_ext_list, dtype=torch.int32, device=_DEVICE)

    total_verify = 0
    for rpi, pfx in zip(fb_rpi_list, fb_pfx_list):
        if rpi == 0:
            continue
        if swa_window_size > 0:
            window_start = max(0, pfx - swa_window_size)
            total_verify += max(0, pfx - window_start)
        else:
            total_verify += pfx

    extras_capacity_pre = 256
    extras_slots, extras_positions, extras_prevs, extras_num_valid = (
        _make_extras_for_kind(kind=extras_kind, capacity=extras_capacity_pre, rng=rng)
    )
    extras_count = int(extras_num_valid[0].item())

    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind=capacity_kind,
        total_verify=total_verify,
        extras_count=extras_count,
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
        extras_slot_indices=extras_slots,
        extras_positions=extras_positions,
        extras_prev_slot_indices=extras_prevs,
        extras_num_valid=extras_num_valid,
        extras_count=extras_count,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )


def _make_extras_for_kind(
    *,
    kind: str,
    capacity: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "none":
        n_valid = 0
    elif kind == "few":
        n_valid = rng.randint(1, 6)
    elif kind == "tile_boundary_64":
        n_valid = 64
    elif kind == "many_129":
        n_valid = 129
    else:
        raise ValueError(f"unknown extras kind {kind}")
    n_valid = min(n_valid, capacity)
    slots = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    positions = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    prevs = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    if n_valid > 0:
        slot_pool = rng.sample(range(500, 500 + max(1000, n_valid * 8)), k=n_valid)
        slots[:n_valid] = torch.tensor(slot_pool, dtype=torch.int32, device=_DEVICE)
        pos_list = [rng.randint(0, 0xFFFF) for _ in range(n_valid)]
        positions[:n_valid] = torch.tensor(pos_list, dtype=torch.int32, device=_DEVICE)
        prev_list = [-1] + slot_pool[: n_valid - 1]
        prevs[:n_valid] = torch.tensor(prev_list, dtype=torch.int32, device=_DEVICE)
    num_valid = torch.tensor([n_valid], dtype=torch.int32, device=_DEVICE)
    return slots, positions, prevs, num_valid


def _run_one(inputs: PlanFuzzInputs) -> tuple:
    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=inputs.verify_capacity,
        write_req_capacity=inputs.write_req_capacity,
    )
    extras_tuple = (
        inputs.extras_slot_indices,
        inputs.extras_positions,
        inputs.extras_prev_slot_indices,
        inputs.extras_num_valid,
    )
    _run_both_and_assert_plan_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=inputs.fb_req_pool_indices,
        fb_prefix_lens=inputs.fb_prefix_lens,
        fb_extend_seq_lens=inputs.fb_extend_seq_lens,
        req_to_token=inputs.req_to_token,
        extras=extras_tuple,
        swa_window_size=inputs.swa_window_size,
        full_to_swa_index_mapping=inputs.full_to_swa_index_mapping,
    )
    assert_all_plan_invariants(
        verify_plan=triton_v,
        write_plan=triton_w,
        fb_req_pool_indices=inputs.fb_req_pool_indices,
        fb_prefix_lens=inputs.fb_prefix_lens,
        fb_extend_seq_lens=inputs.fb_extend_seq_lens,
        swa_window_size=inputs.swa_window_size,
        extras_slot_indices=inputs.extras_slot_indices,
        extras_positions=inputs.extras_positions,
        extras_prev_slot_indices=inputs.extras_prev_slot_indices,
        extras_count=inputs.extras_count,
    )
    return triton_v, triton_w


def _check_repro(inputs: PlanFuzzInputs) -> bool:
    try:
        _run_one(inputs)
    except (AssertionError, RuntimeError, ValueError):
        return True
    return False


def _summarize(inputs: PlanFuzzInputs) -> str:
    return (
        f"bs={int(inputs.fb_req_pool_indices.shape[0])} "
        f"swa={inputs.swa_window_size} extras={inputs.extras_count} "
        f"verify_cap={inputs.verify_capacity} write_cap={inputs.write_req_capacity} "
        f"has_lut={inputs.full_to_swa_index_mapping is not None}"
    )


@pytest.mark.parametrize("seed", _FUZZ_SEEDS_PR)
def test_plan_fuzz_full_combo(seed: int) -> None:
    """Multi-dim plan fuzzer: random LUT/rtt/extras/padding/capacity/swa × N iters, byte-equal."""
    rng = random.Random(seed)
    for iteration in range(_FUZZ_ITER_PER_SEED):
        inputs = _draw_random_plan_inputs(rng)
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
