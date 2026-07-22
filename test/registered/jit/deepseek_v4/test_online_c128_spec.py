from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4.online_c128_mtp import (
    _jit_online_c128_mtp_module,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
STATE_DIM = 3 * HEAD_DIM
MAX_VERIFY_TOKENS = 8


def _layout(bs: int, ragged: bool, device: str):
    if not ragged:
        return (
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
            [MAX_VERIFY_TOKENS] * bs,
        )
    verify_lens = torch.tensor([1, 3, 5, 8][:bs], dtype=torch.int32, device=device)
    starts = torch.nn.functional.pad(torch.cumsum(verify_lens, 0)[:-1], (1, 0))
    return verify_lens, starts, verify_lens.cpu().tolist()


def _reference_prefix_states(
    state: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: list[int],
    starts: list[int],
    state_slot_stride: int,
) -> torch.Tensor:
    expected = state.clone()
    for bid, row_len in enumerate(verify_lens):
        seq_before = int(seq_lens[bid])
        pos = seq_before % 128
        if seq_before > 0 and pos != 0:
            run_max = state[bid, :HEAD_DIM].float()
            run_sum = state[bid, HEAD_DIM : 2 * HEAD_DIM].float()
            run_kv = state[bid, 2 * HEAD_DIM :].float()
        else:
            run_max = torch.zeros(HEAD_DIM, device=state.device)
            run_sum = torch.zeros(HEAD_DIM, device=state.device)
            run_kv = torch.zeros(HEAD_DIM, device=state.device)

        for step in range(row_len):
            row = kv_score_input[starts[bid] + step]
            kv = row[:HEAD_DIM]
            score = row[HEAD_DIM:] + ape[pos]
            if pos == 0:
                run_max = score
                run_sum = torch.ones_like(score)
                run_kv = kv
            else:
                new_max = torch.maximum(run_max, score)
                old_scaled = run_sum * torch.exp(run_max - new_max)
                new_exp = torch.exp(score - new_max)
                new_sum = old_scaled + new_exp
                run_kv = (run_kv * old_scaled + kv * new_exp) / new_sum
                run_max = new_max
                run_sum = new_sum

            final_seq = seq_before + step + 1
            if final_seq % 128 != 0:
                slot = bid + (step + 1) * state_slot_stride
                expected[slot, :HEAD_DIM] = run_max
                expected[slot, HEAD_DIM : 2 * HEAD_DIM] = run_sum
                expected[slot, 2 * HEAD_DIM :] = run_kv
            if pos == 127:
                run_max = torch.zeros_like(run_max)
                run_sum = torch.zeros_like(run_sum)
                run_kv = torch.zeros_like(run_kv)
            pos = (pos + 1) % 128
    return expected


@pytest.mark.parametrize("ragged", [False, True])
def test_online_c128_spec_prefix_and_commit(ragged: bool):
    device = get_device()
    bs = 4
    state_slot_stride = 16
    seq_lens = torch.tensor([1025, 1087, 1144, 1151], device=device)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
    verify_lens, extend_start_loc, verify_lens_list = _layout(bs, ragged, device)
    starts = (
        extend_start_loc.cpu().tolist()
        if ragged
        else [i * MAX_VERIFY_TOKENS for i in range(bs)]
    )
    num_tokens = sum(verify_lens_list)

    torch.manual_seed(7)
    state = torch.randn(
        state_slot_stride * (1 + MAX_VERIFY_TOKENS),
        STATE_DIM,
        dtype=torch.float32,
        device=device,
    )
    state[:bs, HEAD_DIM : 2 * HEAD_DIM].abs_().add_(1.0)
    kv_score_input = torch.randn(
        num_tokens, 2 * HEAD_DIM, dtype=torch.float32, device=device
    )
    ape = torch.randn(128, HEAD_DIM, dtype=torch.float32, device=device)
    req_to_token = torch.zeros((state_slot_stride, 1), dtype=torch.int32, device=device)

    expected = _reference_prefix_states(
        state,
        kv_score_input,
        ape,
        seq_lens,
        verify_lens_list,
        starts,
        state_slot_stride,
    )
    module = _jit_online_c128_mtp_module(
        HEAD_DIM, seq_lens.dtype, req_pool_indices.dtype, state.dtype
    )
    module.write_prefix_states(
        kv_score_input,
        seq_lens,
        req_pool_indices,
        verify_lens,
        extend_start_loc,
        req_to_token,
        ape,
        state,
        bs,
        MAX_VERIFY_TOKENS,
        state_slot_stride,
    )
    torch.testing.assert_close(state, expected, atol=2e-4, rtol=2e-4)

    pending = torch.empty(state_slot_stride, dtype=torch.int64, device=device)
    module.mark_pending(seq_lens, req_pool_indices, pending, bs, state_slot_stride)
    accepts = torch.tensor(
        [0, 1, min(3, verify_lens_list[2]), verify_lens_list[3]],
        dtype=seq_lens.dtype,
        device=device,
    )
    cur_seq_lens = seq_lens + accepts
    expected_main = state[:bs].clone()
    for bid, accept in enumerate(accepts.cpu().tolist()):
        if accept > 0 and int(cur_seq_lens[bid]) % 128 != 0:
            expected_main[bid].copy_(state[bid + accept * state_slot_stride])
    module.commit_pending(
        cur_seq_lens,
        req_pool_indices,
        req_to_token,
        pending,
        state,
        bs,
        MAX_VERIFY_TOKENS,
        state_slot_stride,
        state_slot_stride,
    )
    torch.testing.assert_close(state[:bs], expected_main, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
