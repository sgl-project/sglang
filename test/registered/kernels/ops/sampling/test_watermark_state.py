import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.sampling.watermarking.core import (
    WatermarkState,
    normalize_watermark_request,
)
from sglang.srt.sampling.watermarking.detector import hash_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_repeated_context_and_greedy_bypass():
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device=device,
        default_enabled=True,
    )
    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state.init_from_prompt(req_pool_indices, [[10, 11], [20, 21]])
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device),
        top_ks=torch.tensor([64, 1], device=device, dtype=torch.int32),
        top_ps=torch.ones(2, device=device),
        min_ps=torch.zeros(2, device=device),
        max_top_k=64,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=True,
    )

    first_logits = torch.zeros((2, 64), device=device)
    state.force(first_logits, req_pool_indices, sampling_info)

    assert torch.isfinite(first_logits[0]).sum().item() == 1
    assert torch.equal(first_logits[1], torch.zeros(64, device=device))
    assert state.num_watermarked_contexts.tolist() == [1, 0]

    state.append(
        req_pool_indices[:1], torch.tensor([10], device=device, dtype=torch.int32)
    )
    state.append(
        req_pool_indices[:1], torch.tensor([11], device=device, dtype=torch.int32)
    )
    repeated_logits = torch.zeros((2, 64), device=device)
    state.force(repeated_logits, req_pool_indices, sampling_info)

    assert torch.equal(repeated_logits, torch.zeros_like(repeated_logits))
    assert state.num_watermarked_contexts.tolist() == [1, 0]


def test_low_entropy_bypass_does_not_consume_context():
    state = WatermarkState(
        max_num_reqs=1,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device="cuda",
        default_enabled=True,
        max_probability=0.5,
    )
    req_pool_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    state.init_from_prompt(req_pool_indices, [[10, 11]])
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        top_ks=torch.tensor([2], device="cuda", dtype=torch.int32),
        top_ps=torch.ones(1, device="cuda"),
        min_ps=torch.zeros(1, device="cuda"),
        max_top_k=2,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=True,
    )

    low_entropy_logits = torch.full((1, 64), -torch.inf, device="cuda")
    low_entropy_logits[0, :2] = torch.tensor([1.0, 0.0], device="cuda")
    original = low_entropy_logits.clone()
    state.force(low_entropy_logits, req_pool_indices, sampling_info)

    assert torch.equal(low_entropy_logits, original)
    assert state.num_watermarked_contexts[0].item() == 0

    boundary_logits = torch.full((1, 64), -torch.inf, device="cuda")
    boundary_logits[0, :2] = 0
    state.force(boundary_logits, req_pool_indices, sampling_info)

    assert torch.isfinite(boundary_logits).sum().item() == 1
    assert state.num_watermarked_contexts[0].item() == 1


def test_inactive_batch_skips_watermark_state(monkeypatch):
    state = WatermarkState(
        max_num_reqs=1,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device="cuda",
        default_enabled=True,
    )
    req_pool_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    kernel_activity = Mock()
    monkeypatch.setattr(state, "_ensure_selection_buffers", kernel_activity)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        top_ks=torch.tensor([64], device="cuda", dtype=torch.int32),
        top_ps=torch.ones(1, device="cuda"),
        min_ps=torch.zeros(1, device="cuda"),
        max_top_k=64,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=torch.tensor([False], device="cuda"),
        has_watermark_candidates=False,
    )

    state.force(torch.zeros((1, 64), device="cuda"), req_pool_indices, sampling_info)

    kernel_activity.assert_not_called()
    assert state.lengths[0].item() == 0


def test_retracted_request_restores_context_history():
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        key="0123456789abcdef",
        device=device,
        default_enabled=True,
    )
    req_pool_indices = torch.tensor([1], device=device, dtype=torch.int32)
    request = SimpleNamespace(
        retracted_stain=True,
        origin_input_ids=[10, 11],
        output_ids=[10, 11],
        sampling_params=SimpleNamespace(
            watermark=normalize_watermark_request(
                {"enabled": True, "context_window": 2}
            ),
            top_k=64,
        ),
        get_fill_ids=lambda: [10, 11, 10, 11],
    )
    forward_mode = SimpleNamespace(
        is_extend_without_speculative=lambda: True,
        is_mixed=lambda: False,
    )
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        reqs=[request],
        decoding_reqs=None,
    )
    history = state.retracted_context_hashes(batch)
    assert history is not None
    expected_hashes = {
        hash_context([10, 11]),
        hash_context([11, 10]),
    }
    assert {
        value if value >= 0 else value + 2**32 for value in history[0]
    } == expected_hashes
    state.init_from_prompt(
        req_pool_indices,
        state.prompt_tails(batch),
        history,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device=device),
        top_ks=torch.tensor([64], device=device, dtype=torch.int32),
        top_ps=torch.ones(1, device=device),
        min_ps=torch.zeros(1, device=device),
        max_top_k=64,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=True,
    )
    logits = torch.zeros((1, 64), device=device)
    state.force(logits, req_pool_indices, sampling_info)

    assert torch.equal(logits, torch.zeros_like(logits))
    assert state.num_watermarked_contexts[1].item() == 2


def test_speculative_record_stops_at_context_capacity():
    state = WatermarkState(
        max_num_reqs=1,
        context_window=2,
        max_contexts_per_req=2,
        key="0123456789abcdef",
        device="cuda",
        default_enabled=True,
    )
    state.num_watermarked_contexts[0] = 1
    state.watermarked_context_hashes[0, 0] = 10

    state.record_speculative(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([20, 30, 40], dtype=torch.int64, device="cuda"),
        torch.ones(3, dtype=torch.bool, device="cuda"),
        torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
    )

    assert state.num_watermarked_contexts[0].item() == 2
    assert state.watermarked_context_hashes[0].tolist() == [10, 20]


def test_speculative_tree_fast_path_matches_tensor_reference():
    draft_token_num = 4
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    prompt_tails = [[10, 11, 12, 13], [20, 21, 22, 23]]
    states = [
        WatermarkState(
            max_num_reqs=4,
            context_window=4,
            max_contexts_per_req=16,
            key="0123456789abcdef",
            key_b="fedcba9876543210",
            mixing_probability=0.5,
            device="cuda",
        )
        for _ in range(2)
    ]
    for state in states:
        state.init_from_prompt(req_pool_indices, prompt_tails)

    draft_tokens = torch.tensor(
        [[30, 31, 31, 33], [40, 41, 41, 43]],
        dtype=torch.int32,
        device="cuda",
    ).flatten()
    positions = torch.tensor(
        [[6, 7, 7, 8], [4, 5, 5, 6]], dtype=torch.int64, device="cuda"
    ).flatten()
    tree_rows = (
        (True, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (True, True, False, True),
    )
    custom_mask = torch.tensor(
        [
            value
            for prefix_length in (6, 4)
            for row in tree_rows
            for value in (True,) * prefix_length + row
        ],
        dtype=torch.bool,
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.3]], device="cuda"),
        top_ks=torch.tensor([20, 31], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.95, 0.8], device="cuda"),
        min_ps=torch.tensor([0.0, 0.05], device="cuda"),
        max_top_k=31,
        watermark_keys=torch.tensor(
            [0x0123456789ABCDEF, 0x1111222233334444],
            dtype=torch.int64,
            device="cuda",
        ),
        watermark_context_windows=torch.tensor(
            [4, 3], dtype=torch.int32, device="cuda"
        ),
        watermark_enabled=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    logits = torch.randn(
        (8, 257), generator=torch.Generator("cuda").manual_seed(9), device="cuda"
    )
    reference_logits = logits.clone()
    contexts, context_lengths = states[0].speculative_contexts(
        req_pool_indices,
        draft_tokens,
        custom_mask,
        positions,
        draft_token_num,
        True,
        sampling_info.watermark_context_windows,
    )
    reference_hashes, reference_selected = states[0].force_speculative(
        reference_logits,
        req_pool_indices,
        contexts,
        context_lengths,
        sampling_info,
        draft_token_num,
    )
    fused_logits = logits.clone()
    fused_hashes, fused_selected = states[1].force_speculative_from_tree(
        fused_logits,
        req_pool_indices,
        draft_tokens,
        custom_mask,
        positions,
        sampling_info,
        draft_token_num,
        True,
    )

    assert torch.equal(fused_hashes, reference_hashes)
    assert torch.equal(fused_selected, reference_selected)
    assert torch.equal(fused_logits, reference_logits)
    assert fused_selected.view(2, draft_token_num)[:, 1].all()
    assert fused_selected.view(2, draft_token_num)[:, 2].logical_not().all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
