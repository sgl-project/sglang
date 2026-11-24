import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.verify_accept import VerifyAccept
from sglang.srt.sampling.watermarking.config import (
    normalize_watermark_request,
    parse_watermark_key,
)
from sglang.srt.sampling.watermarking.gumbel.adapter import GumbelEngine, GumbelRows
from sglang.srt.sampling.watermarking.gumbel.detector import hash_context
from sglang.srt.sampling.watermarking.gumbel.state import WatermarkState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_FEATURES = SimpleNamespace(
    watermark_key="0123456789abcdef",
    watermark_default_enabled=True,
    watermark_enforce_all=False,
)


def _server_rows(batch_size, context_window, device="cuda"):
    return GumbelRows(
        keys=torch.full(
            (batch_size,),
            parse_watermark_key(_FEATURES.watermark_key),
            dtype=torch.int64,
            device=device,
        ),
        context_windows=torch.full(
            (batch_size,), context_window, dtype=torch.int32, device=device
        ),
        enabled=torch.ones(batch_size, dtype=torch.bool, device=device),
        candidates_host=[True] * batch_size,
    )


def _force(state, logits, req_pool_indices, sampling_info, rows):
    state.force(
        logits,
        req_pool_indices,
        sampling_info,
        rows.keys,
        rows.context_windows,
        rows.enabled,
    )


def test_repeated_context_and_greedy_bypass():
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        device=device,
    )
    engine = GumbelEngine(state, _FEATURES)
    req_pool_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device),
        top_ks=torch.tensor([64, 1], device=device, dtype=torch.int32),
        top_ps=torch.ones(2, device=device),
        min_ps=torch.zeros(2, device=device),
        max_top_k=64,
        watermark=_server_rows(2, 2),
    )

    def forward_batch(watermark_seed=None):
        return SimpleNamespace(
            sampling_info=sampling_info,
            req_pool_indices=req_pool_indices,
            watermark_seed=watermark_seed,
        )

    first_logits = torch.zeros((2, 64), device=device)
    step = engine.bind_sample(forward_batch(([[10, 11], [20, 21]], None))).apply(
        first_logits, keep_logits=True
    )

    assert torch.equal(step.logprob_logits, torch.zeros_like(first_logits))
    assert torch.isfinite(first_logits[0]).sum().item() == 1
    assert torch.equal(first_logits[1], torch.zeros(64, device=device))
    assert state.num_watermarked_contexts.tolist() == [1, 0]

    # Row 0 samples 10 then 11, back to the context it was forced in.
    for token in (10, 11):
        engine.bind_sample(forward_batch()).commit(
            torch.tensor([token], device=device, dtype=torch.int32)
        )
    repeated_logits = torch.zeros((2, 64), device=device)
    engine.bind_sample(forward_batch()).apply(repeated_logits, keep_logits=False)

    assert torch.equal(repeated_logits, torch.zeros_like(repeated_logits))
    assert state.num_watermarked_contexts.tolist() == [1, 0]


def test_low_entropy_bypass_does_not_consume_context():
    state = WatermarkState(
        max_num_reqs=1,
        context_window=2,
        max_contexts_per_req=8,
        device="cuda",
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
    )

    low_entropy_logits = torch.full((1, 64), -torch.inf, device="cuda")
    low_entropy_logits[0, :2] = torch.tensor([1.0, 0.0], device="cuda")
    original = low_entropy_logits.clone()
    rows = _server_rows(1, 2)
    _force(state, low_entropy_logits, req_pool_indices, sampling_info, rows)

    assert torch.equal(low_entropy_logits, original)
    assert state.num_watermarked_contexts[0].item() == 0

    boundary_logits = torch.full((1, 64), -torch.inf, device="cuda")
    boundary_logits[0, :2] = 0
    _force(state, boundary_logits, req_pool_indices, sampling_info, rows)

    assert torch.isfinite(boundary_logits).sum().item() == 1
    assert state.num_watermarked_contexts[0].item() == 1


def test_retracted_request_restores_context_history():
    device = "cuda"
    state = WatermarkState(
        max_num_reqs=2,
        context_window=2,
        max_contexts_per_req=8,
        device=device,
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
        sampling_info=SimpleNamespace(watermark=_server_rows(1, 2)),
    )
    prompt_tails, history = GumbelEngine(state, _FEATURES).seed_payload(batch)
    assert history is not None
    expected_hashes = {
        hash_context([10, 11]),
        hash_context([11, 10]),
    }
    assert {
        value if value >= 0 else value + 2**32 for value in history[0]
    } == expected_hashes
    state.init_from_prompt(req_pool_indices, prompt_tails, history)
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device=device),
        top_ks=torch.tensor([64], device=device, dtype=torch.int32),
        top_ps=torch.ones(1, device=device),
        min_ps=torch.zeros(1, device=device),
        max_top_k=64,
    )
    logits = torch.zeros((1, 64), device=device)
    _force(state, logits, req_pool_indices, sampling_info, _server_rows(1, 2))

    assert torch.equal(logits, torch.zeros_like(logits))
    assert state.num_watermarked_contexts[1].item() == 2


def test_speculative_record_stops_at_context_capacity():
    state = WatermarkState(
        max_num_reqs=1,
        context_window=2,
        max_contexts_per_req=2,
        device="cuda",
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


def test_speculative_tree_paths_match_tensor_reference():
    draft_token_num = 4
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    prompt_tails = [[10, 11, 12, 13], [20, 21, 22, 23]]
    states = [
        WatermarkState(
            max_num_reqs=4,
            context_window=4,
            max_contexts_per_req=16,
            key_b="fedcba9876543210",
            mixing_probability=0.5,
            device="cuda",
        )
        for _ in range(3)
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
    )
    keys = torch.tensor(
        [0x0123456789ABCDEF, 0x1111222233334444],
        dtype=torch.int64,
        device="cuda",
    )
    context_windows = torch.tensor([4, 3], dtype=torch.int32, device="cuda")
    watermark_enabled = torch.ones(2, dtype=torch.bool, device="cuda")
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
        context_windows,
    )
    reference_hashes, reference_selected = states[0].force_speculative(
        reference_logits,
        req_pool_indices,
        contexts,
        context_lengths,
        sampling_info,
        draft_token_num,
        keys,
        watermark_enabled,
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
        keys,
        context_windows,
        watermark_enabled,
    )

    assert torch.equal(fused_hashes, reference_hashes)
    assert torch.equal(fused_selected, reference_selected)
    assert torch.equal(fused_logits, reference_logits)
    assert fused_selected.view(2, draft_token_num)[:, 1].all()
    assert fused_selected.view(2, draft_token_num)[:, 2].logical_not().all()

    rows = GumbelRows(
        keys=keys,
        context_windows=context_windows,
        enabled=watermark_enabled,
        candidates_host=[True, True],
    )
    verify = GumbelEngine(states[2], _FEATURES).bind_verify(rows, full_mask=True)
    verify_logits, _ = verify.apply(
        logits,
        sampling_info=sampling_info,
        req_pool_indices=req_pool_indices,
        draft_tokens=draft_tokens,
        positions=positions,
        draft_token_num=draft_token_num,
        custom_mask=custom_mask,
        keep_logits=True,
    )
    assert torch.equal(verify_logits, reference_logits)
    assert torch.isfinite(logits).all()

    # Accepted paths 0 -> 1 -> 3 and 4 -> 5; row 1's 7 is past its length.
    accept_index = torch.tensor(
        [[0, 1, 3], [4, 5, 7]], dtype=torch.int32, device="cuda"
    )
    accept_lens = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    verify.commit(
        VerifyAccept(
            tokens=draft_tokens, accept_lens=accept_lens, accept_index=accept_index
        )
    )
    states[0].record_speculative(
        req_pool_indices,
        reference_hashes,
        reference_selected,
        accept_index,
        accept_lens,
    )
    assert (states[2].num_watermarked_contexts[req_pool_indices] > 0).all()
    assert torch.equal(
        states[2].num_watermarked_contexts, states[0].num_watermarked_contexts
    )
    # Slots past each row's count are uninitialized (torch.empty).
    for row, count in enumerate(states[0].num_watermarked_contexts.tolist()):
        assert torch.equal(
            states[2].watermarked_context_hashes[row, :count],
            states[0].watermarked_context_hashes[row, :count],
        )


@pytest.mark.parametrize("block_verification", [False, True])
def test_dspark_chain_commits_non_speculative_contexts(block_verification):
    from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptSampling

    vocab_size = 64
    req_pool_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        top_ks=torch.tensor([vocab_size], dtype=torch.int32, device="cuda"),
        top_ps=torch.ones(1, device="cuda"),
        min_ps=torch.zeros(1, device="cuda"),
        max_top_k=vocab_size,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    states = [
        WatermarkState(
            max_num_reqs=1,
            context_window=2,
            max_contexts_per_req=8,
            device="cuda",
        )
        for _ in range(2)
    ]
    reference, state = states
    for each in states:
        each.init_from_prompt(req_pool_indices, [[10, 11]])
    logits = torch.randn(
        (4, vocab_size), generator=torch.Generator("cuda").manual_seed(3), device="cuda"
    )

    forced = []
    rows = _server_rows(1, 2)
    for row in range(2):
        row_logits = logits[row : row + 1].clone()
        _force(reference, row_logits, req_pool_indices, sampling_info, rows)
        forced.append(int(torch.isfinite(row_logits[0]).nonzero().item()))
        reference.append(
            req_pool_indices,
            torch.tensor([forced[-1]], dtype=torch.int32, device="cuda"),
        )

    chain_tokens = torch.tensor(
        [[11, forced[0], (forced[1] + 1) % vocab_size, 7]],
        dtype=torch.int64,
        device="cuda",
    )
    engine = GumbelEngine(state, _FEATURES)
    verify = engine.bind_verify(rows, full_mask=False)
    target_logits, _ = verify.apply(
        logits.clone(),
        sampling_info=sampling_info,
        req_pool_indices=req_pool_indices,
        draft_tokens=chain_tokens,
        positions=torch.arange(4, device="cuda").view(1, 4),
        draft_token_num=4,
        custom_mask=None,
        keep_logits=False,
    )
    draft_probs = torch.softmax(
        torch.randn(
            (1, 3, vocab_size),
            generator=torch.Generator("cuda").manual_seed(5),
            device="cuda",
        ),
        dim=-1,
    )
    correct_len, bonus, _ = AcceptSampling.execute(
        candidates=chain_tokens,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
        gamma=3,
        verify_num_draft_tokens=4,
        block_verification=block_verification,
    )
    assert correct_len.tolist() == [1]
    assert bonus.tolist() == [forced[1]]

    accept_lens = torch.tensor([2], dtype=torch.int32, device="cuda")
    verify.commit(
        VerifyAccept(
            tokens=torch.tensor(
                [[forced[0], forced[1], 0, 0]], dtype=torch.int64, device="cuda"
            ),
            accept_lens=accept_lens,
        )
    )
    assert torch.equal(
        state.contexts_tail(req_pool_indices)[0],
        reference.contexts_tail(req_pool_indices)[0],
    )
    assert state.num_watermarked_contexts.tolist() == [2]
    assert torch.equal(
        state.watermarked_context_hashes[0, :2],
        reference.watermarked_context_hashes[0, :2],
    )


def test_chain_commit_matches_explicit_accept_index():
    draft_token_num = 4
    req_pool_indices = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), device="cuda"),
        top_ks=torch.tensor([64, 64], dtype=torch.int32, device="cuda"),
        top_ps=torch.ones(2, device="cuda"),
        min_ps=torch.zeros(2, device="cuda"),
        max_top_k=64,
    )
    draft_tokens = torch.tensor(
        [[30, 31, 32, 33], [40, 41, 42, 43]], dtype=torch.int64, device="cuda"
    )
    logits = torch.randn(
        (2 * draft_token_num, 64),
        generator=torch.Generator("cuda").manual_seed(7),
        device="cuda",
    )
    rows = _server_rows(2, 2)
    states = []
    for accept_index in (
        None,
        torch.arange(2 * draft_token_num, dtype=torch.int32, device="cuda").view(
            2, draft_token_num
        ),
    ):
        state = WatermarkState(
            max_num_reqs=3, context_window=2, max_contexts_per_req=8, device="cuda"
        )
        state.init_from_prompt(req_pool_indices, [[10, 11], [20, 21]])
        verify = GumbelEngine(state, _FEATURES).bind_verify(rows, full_mask=False)
        verify.apply(
            logits.clone(),
            sampling_info=sampling_info,
            req_pool_indices=req_pool_indices,
            draft_tokens=draft_tokens,
            positions=torch.arange(2, 6, device="cuda").repeat(2, 1),
            draft_token_num=draft_token_num,
            custom_mask=None,
            keep_logits=False,
        )
        verify.commit(
            VerifyAccept(
                tokens=draft_tokens if accept_index is None else draft_tokens.view(-1),
                accept_lens=torch.tensor([3, 2], dtype=torch.int32, device="cuda"),
                accept_index=accept_index,
            )
        )
        states.append(state)

    for chain, indexed in zip(
        states[0].contexts_tail(req_pool_indices),
        states[1].contexts_tail(req_pool_indices),
    ):
        assert torch.equal(chain, indexed)
    counts = states[1].num_watermarked_contexts.tolist()
    assert counts == [2, 0, 3]
    assert states[0].num_watermarked_contexts.tolist() == counts
    for row, count in enumerate(counts):
        assert torch.equal(
            states[0].watermarked_context_hashes[row, :count],
            states[1].watermarked_context_hashes[row, :count],
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
