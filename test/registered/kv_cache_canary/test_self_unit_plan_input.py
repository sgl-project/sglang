from __future__ import annotations

import torch

from sglang.srt.kv_cache_canary.plan_input import (
    AliveReqSnapshot,
    build_plan_input_per_forward,
    build_plan_input_radix_sweep,
    build_plan_input_running_sweep,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def test_build_plan_input_per_forward_extend(device, make_forward_batch):
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([10, 12], dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor([3, 5], dtype=torch.int32, device=device),
        extend_seq_lens=torch.tensor([7, 7], dtype=torch.int32, device=device),
        is_extend=True,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, swa_window_size=0, full_to_swa_index_mapping=None
    )
    assert out is not None
    assert out.fb_prefix_lens.tolist() == [3, 5]
    assert out.fb_extend_seq_lens.tolist() == [7, 7]
    assert out.fb_req_pool_indices.data_ptr() == fb.req_pool_indices.data_ptr()


def test_build_plan_input_per_forward_decode(device, make_forward_batch):
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([4, 7, 1], dtype=torch.int32, device=device),
        is_extend=False,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, swa_window_size=0, full_to_swa_index_mapping=None
    )
    assert out is not None
    assert out.fb_prefix_lens.tolist() == [3, 6, 0]
    assert out.fb_extend_seq_lens.tolist() == [1, 1, 1]


def test_build_plan_input_running_sweep(device, make_req_to_token_pool):
    pool = make_req_to_token_pool()
    rpi = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([4, 5, 6], dtype=torch.int32, device=device)
    snapshot = AliveReqSnapshot(req_pool_indices=rpi, seq_lens=seq_lens)
    out = build_plan_input_running_sweep(
        req_to_token_pool=pool,
        alive_reqs=snapshot,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    assert out.fb_req_pool_indices.tolist() == [1, 2, 3]
    assert out.fb_prefix_lens.tolist() == [4, 5, 6]
    assert out.fb_extend_seq_lens.tolist() == [0, 0, 0]
    assert int(out.extra_verify_num_valid.item()) == 0


def test_build_plan_input_radix_sweep(device, make_radix_cache, make_req_to_token_pool):
    empty_cache = make_radix_cache([[]])
    empty_cache.req_to_token_pool = make_req_to_token_pool()
    empty_out = build_plan_input_radix_sweep(
        radix_cache=empty_cache,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    assert int(empty_out.extra_verify_num_valid.item()) == 0

    cache = make_radix_cache([[], [100, 101, 102]])
    cache.req_to_token_pool = make_req_to_token_pool()
    out = build_plan_input_radix_sweep(
        radix_cache=cache,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    assert int(out.extra_verify_num_valid.item()) == 3
    assert out.extra_verify_slot_indices[:3].tolist() == [100, 101, 102]
    assert out.extra_verify_positions[:3].tolist() == [0, 1, 2]
    assert out.extra_verify_prev_slot_indices[:3].tolist() == [-1, 100, 101]


def test_plan_input_padding_dummy_sentinel(device, make_forward_batch):
    fb = make_forward_batch(
        req_pool_indices=torch.tensor([0, 5, 0], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([0, 3, 0], dtype=torch.int32, device=device),
        is_extend=False,
    )
    out = build_plan_input_per_forward(
        forward_batch=fb, swa_window_size=0, full_to_swa_index_mapping=None
    )
    assert out is not None
    assert out.fb_req_pool_indices.tolist() == [0, 5, 0]
    assert out.fb_req_pool_indices.dtype == torch.int64


def test_radix_held_slot_still_swept(device, make_radix_cache, make_req_to_token_pool):
    cache = make_radix_cache([[], [42, 43, 44]])
    cache.req_to_token_pool = make_req_to_token_pool()
    out = build_plan_input_radix_sweep(
        radix_cache=cache,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    n = int(out.extra_verify_num_valid.item())
    assert n == 3
    assert set(out.extra_verify_slot_indices[:n].tolist()) == {42, 43, 44}


def test_truly_free_slot_not_swept(device, make_radix_cache, make_req_to_token_pool):
    empty_cache = make_radix_cache([[]])
    empty_cache.req_to_token_pool = make_req_to_token_pool()
    out = build_plan_input_radix_sweep(
        radix_cache=empty_cache,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    assert int(out.extra_verify_num_valid.item()) == 0
