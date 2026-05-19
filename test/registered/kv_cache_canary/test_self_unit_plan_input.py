from __future__ import annotations

import torch

from sglang.srt.kv_cache_canary.plan_input import (
    build_plan_input_per_forward,
    build_plan_input_radix_sweep,
    build_plan_input_running_sweep,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def test_build_plan_input_per_forward_extend(
    device, make_forward_batch, make_req_to_token_pool
):
    pool = make_req_to_token_pool()
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([10, 12], dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor([3, 5], dtype=torch.int32, device=device),
        extend_seq_lens=torch.tensor([7, 7], dtype=torch.int32, device=device),
        is_extend=True,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, req_to_token_pool=pool, extras_capacity=1
    )
    assert out is not None
    assert out.fb_prefix_lens.tolist() == [3, 5]
    assert out.fb_extend_seq_lens.tolist() == [7, 7]
    assert out.req_to_token.data_ptr() == pool.req_to_token.data_ptr()


def test_build_plan_input_per_forward_decode(
    device, make_forward_batch, make_req_to_token_pool
):
    pool = make_req_to_token_pool()
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([4, 7, 1], dtype=torch.int32, device=device),
        is_extend=False,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, req_to_token_pool=pool, extras_capacity=1
    )
    assert out is not None
    assert out.fb_prefix_lens.tolist() == [3, 6, 0]
    assert out.fb_extend_seq_lens.tolist() == [1, 1, 1]


def test_build_plan_input_running_sweep(device, make_req_to_token_pool):
    pool = make_req_to_token_pool()
    rpi = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([4, 5, 6], dtype=torch.int32, device=device)
    out = build_plan_input_running_sweep(
        req_to_token_pool=pool,
        running_req_pool_indices=rpi,
        running_seq_lens=seq_lens,
        extras_capacity=1,
    )
    assert out.fb_req_pool_indices.tolist() == [1, 2, 3]
    assert out.fb_prefix_lens.tolist() == [4, 5, 6]
    assert out.fb_extend_seq_lens.tolist() == [0, 0, 0]
    assert int(out.extra_verify_num_valid.item()) == 0


def test_build_plan_input_radix_sweep(device, make_req_to_token_pool, make_radix_cache):
    pool = make_req_to_token_pool()
    empty_cache = make_radix_cache([[]])
    empty_out = build_plan_input_radix_sweep(
        req_to_token_pool=pool, radix_cache=empty_cache, extras_capacity=4
    )
    assert int(empty_out.extra_verify_num_valid.item()) == 0

    cache = make_radix_cache([[], [100, 101, 102]])
    out = build_plan_input_radix_sweep(
        req_to_token_pool=pool, radix_cache=cache, extras_capacity=8
    )
    assert int(out.extra_verify_num_valid.item()) == 3
    assert out.extra_verify_slot_indices[:3].tolist() == [100, 101, 102]
    assert out.extra_verify_positions[:3].tolist() == [0, 1, 2]
    assert out.extra_verify_prev_slot_indices[:3].tolist() == [-1, 100, 101]


def test_plan_input_padding_dummy_sentinel(
    device, make_forward_batch, make_req_to_token_pool
):
    pool = make_req_to_token_pool()
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([0, 5, 0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([0, 3, 0], dtype=torch.int32, device=device),
        is_extend=False,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, req_to_token_pool=pool, extras_capacity=1
    )
    assert out is not None
    assert out.fb_req_pool_indices.tolist() == [0, 5, 0]
    assert out.fb_req_pool_indices.dtype == torch.int32


def test_radix_held_slot_still_swept(device, make_req_to_token_pool, make_radix_cache):
    pool = make_req_to_token_pool()
    cache = make_radix_cache([[], [42, 43, 44]])
    out = build_plan_input_radix_sweep(
        req_to_token_pool=pool, radix_cache=cache, extras_capacity=8
    )
    n = int(out.extra_verify_num_valid.item())
    assert n == 3
    assert set(out.extra_verify_slot_indices[:n].tolist()) == {42, 43, 44}


def test_truly_free_slot_not_swept(device, make_req_to_token_pool, make_radix_cache):
    pool = make_req_to_token_pool()
    empty_cache = make_radix_cache([[]])
    out = build_plan_input_radix_sweep(
        req_to_token_pool=pool, radix_cache=empty_cache, extras_capacity=4
    )
    assert int(out.extra_verify_num_valid.item()) == 0
