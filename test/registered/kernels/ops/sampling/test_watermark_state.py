import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.sampling.watermark import (
    WatermarkState,
    _dual_key_a_mask_torch,
    _hash_context_token_ids,
    _hash_contexts,
    _truncate_probabilities,
    normalize_watermark_request,
    select_watermark_tokens_torch,
)
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


@pytest.mark.parametrize("enabled,top_k", [(False, 64), (True, 1)])
def test_inactive_batch_skips_watermark_state(monkeypatch, enabled, top_k):
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
        top_ks=torch.tensor([top_k], device="cuda", dtype=torch.int32),
        top_ps=torch.ones(1, device="cuda"),
        min_ps=torch.zeros(1, device="cuda"),
        max_top_k=top_k,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=torch.tensor([enabled], device="cuda"),
        has_watermark_candidates=False,
    )

    state.init_from_prompt(req_pool_indices, [[10, 11]], active=False)
    state.force(torch.zeros((1, 64), device="cuda"), req_pool_indices, sampling_info)
    state.append(
        req_pool_indices,
        torch.tensor([12], device="cuda", dtype=torch.int32),
        active=False,
    )

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
        _hash_context_token_ids([10, 11]),
        _hash_context_token_ids([11, 10]),
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


def test_dual_key_speculative_rows_match_per_request_config():
    draft_token_num = 3
    state = WatermarkState(
        max_num_reqs=4,
        context_window=4,
        max_contexts_per_req=16,
        key="0123456789abcdef",
        key_b="fedcba9876543210",
        mixing_probability=0.5,
        device="cuda",
    )
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    state.key_b_buffer[1] = 0x2222333344445555
    state.key_b_buffer[3] = 0x3333444455556666
    state.mixing_threshold_buffer[1] = 1 << 30
    state.mixing_threshold_buffer[3] = 3 << 30
    contexts = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [11, 12, 13, 14],
            [12, 13, 14, 15],
            [13, 14, 15, 16],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    context_lengths = torch.full((6,), 4, dtype=torch.int32, device="cuda")
    request_keys = torch.tensor(
        [0x0123456789ABCDEF, 0x1111222233334444],
        dtype=torch.int64,
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.7], [1.3]], device="cuda"),
        top_ks=torch.tensor([17, 31], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.95, 0.8], device="cuda"),
        min_ps=torch.tensor([0.0, 0.05], device="cuda"),
        max_top_k=31,
        watermark_keys=request_keys,
        watermark_context_windows=torch.full((2,), 4, dtype=torch.int32, device="cuda"),
        watermark_enabled=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    generator = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn((6, 257), generator=generator, device="cuda")
    expected_logits = logits.clone()
    context_hashes = _hash_contexts(contexts, context_lengths)
    expanded_keys = request_keys.repeat_interleave(draft_token_num)
    pool_indices = req_pool_indices.to(torch.int64)
    expanded_keys_b = state.key_b_buffer[pool_indices].repeat_interleave(
        draft_token_num
    )
    expanded_thresholds = state.mixing_threshold_buffer[pool_indices].repeat_interleave(
        draft_token_num
    )
    key_a_mask = _dual_key_a_mask_torch(
        expanded_keys, expanded_keys_b, context_hashes, expanded_thresholds
    )
    assert key_a_mask.any() and key_a_mask.logical_not().any()
    assert torch.equal(
        expanded_keys,
        torch.tensor(
            [
                0x0123456789ABCDEF,
                0x0123456789ABCDEF,
                0x0123456789ABCDEF,
                0x1111222233334444,
                0x1111222233334444,
                0x1111222233334444,
            ],
            dtype=torch.int64,
            device="cuda",
        ),
    )
    expanded_temperatures = sampling_info.temperatures.repeat_interleave(
        draft_token_num, dim=0
    )
    expanded_top_ks = sampling_info.top_ks.repeat_interleave(draft_token_num)
    expanded_top_ps = sampling_info.top_ps.repeat_interleave(draft_token_num)
    expanded_min_ps = sampling_info.min_ps.repeat_interleave(draft_token_num)
    probabilities = _truncate_probabilities(
        expected_logits,
        expanded_temperatures,
        expanded_top_ks,
        expanded_top_ps,
        expanded_min_ps,
    )
    expected_tokens = select_watermark_tokens_torch(
        probabilities.float(),
        context_hashes,
        expanded_keys,
        expanded_keys_b,
        expanded_thresholds,
    )
    expected_logits.fill_(-torch.inf)
    expected_logits[torch.arange(6, device="cuda"), expected_tokens.to(torch.int64)] = 0

    actual_hashes, selected = state.force_speculative(
        logits,
        req_pool_indices,
        contexts,
        context_lengths,
        sampling_info,
        draft_token_num,
    )

    assert selected.all()
    assert torch.equal(actual_hashes, context_hashes)
    assert torch.equal(logits, expected_logits)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
