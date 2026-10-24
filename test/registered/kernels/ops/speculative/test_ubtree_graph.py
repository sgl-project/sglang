# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See python/sglang/srt/speculative/ubtree_components/LICENSE.

"""UBTree proposal, verify metadata, and cache-commit CUDA Graph regressions."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config
from sglang.srt.speculative.ubtree_components.epilogue import UBTreeVerifyEpilogue
from sglang.srt.speculative.ubtree_components.selector import UBTreeSelector
from sglang.srt.speculative.ubtree_components.tree_graph import (
    build_ubtree_proposal,
)
from sglang.srt.speculative.ubtree_info import UBTreeVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _config():
    return SimpleNamespace(
        vocab_size=17,
        hidden_size=8,
        block_size=5,
        ubtree_config={
            "rank": 4,
            "candidate_k": 3,
            "tree_budget": 5,
            "tree_supertree_width": 3,
            "factorizer_hidden_width": 6,
            "target_embedding_dim": 8,
            "depth_condition_count": 4,
            "objective": "kl",
            "adds_unary_logits": True,
            "token_factorizer": "target_embedding_mlp",
            "hidden_projection_type": "shared",
            "tree_depth_bonus": 0.0,
        },
    )


def _lookup_config():
    config = _config()
    config.ubtree_config = {
        **config.ubtree_config,
        "token_factorizer": "lookup",
        "factorizer_hidden_width": 0,
        "target_embedding_dim": 0,
    }
    return config


def _make_selector_graph_components(device):
    config = _config()
    selector = UBTreeSelector(config).to(device=device, dtype=torch.float32)
    embedding = torch.randn(config.vocab_size, config.hidden_size, device=device)
    selector.materialize_cache(embedding)
    lm_head = torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False, device=device
    )
    return config, selector, lm_head


def _assert_tree_batches_equal(actual, expected):
    assert actual.__dict__.keys() == expected.__dict__.keys()
    for name in actual.__dict__:
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if expected_value is None or actual_value is None:
            assert actual_value is expected_value, name
        elif isinstance(expected_value, torch.Tensor):
            if expected_value.dtype.is_floating_point:
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert torch.equal(actual_value, expected_value), name
        else:
            assert actual_value == expected_value, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_proposal_tail_replays_changing_inputs_exactly():
    config, selector, lm_head = _make_selector_graph_components("cuda")
    hidden = torch.zeros(2, 4, 8, device="cuda")
    bonus_tokens = torch.zeros(2, dtype=torch.int64, device="cuda")

    def build():
        return build_ubtree_proposal(
            hidden=hidden,
            bonus_tokens=bonus_tokens,
            config=parse_ubtree_config(config),
            num_nodes=6,
            lm_head=lm_head,
            selector=selector,
        )

    for _ in range(2):
        build()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build()
    for seed in (61, 62):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        hidden.copy_(torch.randn(2, 4, 8, device="cuda", generator=generator))
        bonus_tokens.copy_(
            torch.randint(0, 17, (2,), device="cuda", generator=generator)
        )
        expected = build()
        graph.replay()
        torch.cuda.synchronize()
        _assert_tree_batches_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_draft_commit_metadata_is_cuda_graph_replay_safe():
    from sglang.kernels.ops.speculative.ubtree import (
        prepare_draft_commit_metadata,
    )

    device = torch.device("cuda")
    req_to_token = torch.arange(4 * 64, dtype=torch.int64, device=device).view(4, 64)
    seq_lens = torch.tensor([2, 9], dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([3, 1], dtype=torch.int32, device=device)
    positions = torch.empty((2, 16), dtype=torch.int64, device=device)
    cache_locs = torch.empty_like(positions)

    # Compile before capture; the replay itself must only use stable tensor addresses.
    prepare_draft_commit_metadata(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        positions_out=positions,
        cache_loc_out=cache_locs,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_draft_commit_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            positions_out=positions,
            cache_loc_out=cache_locs,
        )

    seq_lens.copy_(torch.tensor([5, 17], dtype=torch.int32, device=device))
    req_pool_indices.copy_(torch.tensor([0, 2], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    offsets = torch.arange(16, dtype=torch.int64, device=device)
    expected_positions = seq_lens.to(torch.int64)[:, None] + offsets
    expected_cache_locs = req_to_token[
        req_pool_indices.to(torch.int64)[:, None], expected_positions
    ]
    assert torch.equal(positions, expected_positions)
    assert torch.equal(cache_locs, expected_cache_locs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_target_kv_move_metadata_is_cuda_graph_replay_safe():
    from sglang.kernels.ops.speculative.ubtree import (
        prepare_target_kv_move_metadata,
    )

    device = torch.device("cuda")
    req_to_token = torch.arange(4 * 64, dtype=torch.int64, device=device).view(4, 64)
    req_pool_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([3, 11], dtype=torch.int32, device=device)
    accept_index = torch.tensor(
        [[0, 2, 4, -1], [5, 6, -1, -1]], dtype=torch.int32, device=device
    )
    accept_lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
    out_cache_loc = torch.arange(10, dtype=torch.int64, device=device) + 1000
    target = torch.empty((8,), dtype=torch.int64, device=device)
    source = torch.empty_like(target)

    kwargs = dict(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        accept_index=accept_index,
        accept_lens=accept_lens,
        out_cache_loc=out_cache_loc,
        target_cache_loc_out=target,
        source_cache_loc_out=source,
    )
    prepare_target_kv_move_metadata(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_target_kv_move_metadata(**kwargs)

    seq_lens.copy_(torch.tensor([7, 13], dtype=torch.int32, device=device))
    req_pool_indices.copy_(torch.tensor([1, 3], dtype=torch.int32, device=device))
    accept_index.copy_(
        torch.tensor([[1, -1, -1, -1], [3, 7, 8, -1]], dtype=torch.int32, device=device)
    )
    accept_lens.copy_(torch.tensor([1, 3], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(target.cpu(), torch.tensor([71, 205, 206, 207, 0, 0, 0, 0]))
    assert torch.equal(source.cpu(), torch.tensor([1001, 1003, 1007, 1008, 0, 0, 0, 0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_unified_commit_metadata_is_cuda_graph_replay_safe():
    from sglang.kernels.ops.speculative.ubtree import (
        prepare_unified_commit_metadata,
    )

    device = torch.device("cuda")
    target_req_to_token = torch.arange(4 * 64, dtype=torch.int64, device=device).view(
        4, 64
    )
    draft_req_to_token = target_req_to_token + 1000
    req_pool_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([3, 11], dtype=torch.int32, device=device)
    accept_index = torch.tensor(
        [[0, 2, 4, -1], [5, 6, -1, -1]], dtype=torch.int32, device=device
    )
    accept_lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
    out_cache_loc = torch.arange(10, dtype=torch.int64, device=device) + 2000
    target = torch.empty((2, 4), dtype=torch.int64, device=device)
    source = torch.empty_like(target)
    positions = torch.empty_like(target)
    draft = torch.empty_like(target)
    kwargs = dict(
        req_pool_indices=req_pool_indices,
        target_req_to_token=target_req_to_token,
        draft_req_to_token=draft_req_to_token,
        seq_lens=seq_lens,
        accept_index=accept_index,
        accept_lens=accept_lens,
        out_cache_loc=out_cache_loc,
        target_cache_loc_out=target,
        source_cache_loc_out=source,
        draft_positions_out=positions,
        draft_cache_loc_out=draft,
    )

    prepare_unified_commit_metadata(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_unified_commit_metadata(**kwargs)

    seq_lens.copy_(torch.tensor([7, 13], dtype=torch.int32, device=device))
    req_pool_indices.copy_(torch.tensor([1, 3], dtype=torch.int32, device=device))
    accept_index.copy_(
        torch.tensor([[1, -1, -1, -1], [3, 7, 8, -1]], dtype=torch.int32, device=device)
    )
    accept_lens.copy_(torch.tensor([1, 3], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(positions.cpu(), torch.tensor([[7, 8, 9, 10], [13, 14, 15, 16]]))
    assert torch.equal(target.cpu(), torch.tensor([[71, 0, 0, 0], [205, 206, 207, 0]]))
    assert torch.equal(
        source.cpu(), torch.tensor([[2001, 0, 0, 0], [2003, 2007, 2008, 0]])
    )
    assert torch.equal(
        draft.cpu(), torch.tensor([[1071, 1072, 1073, 1074], [1205, 1206, 1207, 1208]])
    )


@pytest.mark.parametrize("track", [False, True])
def test_verify_epilogue_folds_factorized_kda_commit(monkeypatch, track):
    epilogue = UBTreeVerifyEpilogue(
        max_bs=2,
        node_count=5,
        max_depth=3,
        device="cpu",
    )
    spec_state = SimpleNamespace(tree_rawv=torch.empty((1, 2, 1, 5, 2)))
    mapping = torch.tensor([7, 11, 13, 17], dtype=torch.int32)
    epilogue.attach_kda_tree_commit(
        spec_state=spec_state,
        mamba_index_mapping=mapping,
        mamba_track_interval=64,
    )
    verify_input = UBTreeVerifyInput(
        draft_token=torch.arange(10, dtype=torch.int64),
        retrieve_index=torch.arange(10, dtype=torch.int64).view(2, 5),
        retrieve_next_token=torch.full((2, 5), -1, dtype=torch.int64),
        retrieve_next_sibling=torch.full((2, 5), -1, dtype=torch.int64),
        custom_mask=None,
        positions=None,
        retrieve_cum_len=None,
        spec_steps=2,
        topk=2,
        draft_token_num=5,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        logical_max_depth=3,
    )
    seq_lens = torch.tensor([60, 20], dtype=torch.int64)
    req_pool_indices = torch.tensor([2, 0], dtype=torch.int64)
    epilogue.begin_step(verify_input, seq_lens, 2)

    def fake_verify(**kwargs):
        kwargs["predicts"].copy_(kwargs["target_predict"].reshape(-1))
        kwargs["accept_index"].copy_(
            torch.tensor([[0, 2, -1], [5, 7, 8]], dtype=torch.int32)
        )
        kwargs["accept_token_num"].copy_(torch.tensor([1, 2], dtype=torch.int32))

    def fake_track(
        accept_index,
        accept_lens,
        seq_lens,
        draft_token_num,
        mamba_track_interval,
        *,
        last_correct_out,
        track_steps_out,
    ):
        assert track, "tracking indices must not be computed without tracking"
        last_correct_out[:2].copy_(torch.tensor([2, 3], dtype=torch.int64))
        track_steps_out[:2].copy_(torch.tensor([0, -1], dtype=torch.int64))
        return last_correct_out[:2], track_steps_out[:2]

    committed = {}

    def fake_commit(**kwargs):
        committed.update(
            {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
        )

    monkeypatch.setattr(
        "sglang.srt.speculative.ubtree_components.epilogue.verify_tree_greedy_func",
        fake_verify,
    )
    monkeypatch.setattr(
        "sglang.srt.speculative.ubtree_components.epilogue.fused_commit_track_indices",
        fake_track,
    )
    monkeypatch.setattr(
        "sglang.srt.speculative.ubtree_components.epilogue.commit_kda_tree_after_verify",
        fake_commit,
    )

    logits = torch.full((10, 16), -10.0)
    logits[torch.arange(10), torch.arange(10)] = 10.0
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    track_indices = torch.tensor([21, 22], dtype=torch.int64) if track else None
    epilogue(
        logits=logits,
        hidden=hidden,
        bs=2,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        mamba_track_indices=track_indices,
    )

    assert epilogue.folds_kda_commit
    assert torch.equal(
        committed["state_batch_indices"], torch.tensor([13, 7], dtype=torch.int32)
    )
    assert torch.equal(
        committed["factor_cache_indices"], torch.tensor([0, 1], dtype=torch.int32)
    )
    assert torch.equal(
        committed["accept_lens"], torch.tensor([2, 3], dtype=torch.int32)
    )
    if track:
        assert torch.equal(committed["mamba_track_indices"], track_indices)
        assert torch.equal(committed["mamba_steps_to_track"], torch.tensor([0, -1]))
    else:
        assert committed["mamba_track_indices"] is None
        assert committed["mamba_steps_to_track"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("graph_replay", [False, True])
def test_worker_commits_once_and_keeps_scheduler_outputs_alive(
    monkeypatch, graph_replay
):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import ubtree_worker_v2 as worker_module
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
    from sglang.srt.speculative.ubtree_components.tree import build_ubtree_batch

    class CountingEpilogue(UBTreeVerifyEpilogue):
        calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            return super().__call__(**kwargs)

    worker = worker_module.UBTreeWorkerV2.__new__(worker_module.UBTreeWorkerV2)
    worker.device = "cuda"
    worker.ubtree_config = SimpleNamespace(block_size=3, depth_count=2)
    worker._verify_positions_buf = torch.empty((1, 5), dtype=torch.int64, device="cuda")
    epilogue = worker._verify_epilogue = CountingEpilogue(
        max_bs=1, node_count=5, max_depth=3, device="cuda"
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        spec_info=DFlashDraftInputV2.create_idle_input(device="cuda"),
        seq_lens=torch.tensor([3], dtype=torch.int32, device="cuda"),
    )
    tree = build_ubtree_batch(
        anchor_tokens=torch.tensor([0], device="cuda"),
        candidate_ids=torch.tensor([[[1, 2], [3, 4]]], device="cuda"),
        edge_log_probs=torch.log_softmax(torch.zeros((1, 2, 2, 2), device="cuda"), -1),
        budget=4,
        supertree_width=2,
    )
    worker._draft_tree = lambda *args: (tree, batch.seq_lens, None)
    worker.req_to_token_pool = None
    forward = SimpleNamespace(
        req_pool_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        seq_lens=batch.seq_lens,
        out_cache_loc=torch.arange(5, device="cuda"),
        mamba_track_indices=None,
    )
    logits = torch.zeros((5, 17), device="cuda")
    logits[:, 11] = 10
    output = LogitsProcessorOutput(
        next_token_logits=logits, hidden_states=torch.randn((5, 8), device="cuda")
    )

    def target_forward(**kwargs):
        if graph_replay:
            epilogue(
                logits=logits,
                hidden=output.hidden_states,
                bs=1,
                req_pool_indices=forward.req_pool_indices,
                seq_lens=forward.seq_lens,
                out_cache_loc=forward.out_cache_loc,
            )
        return SimpleNamespace(logits_output=output, can_run_cuda_graph=graph_replay)

    worker._target_worker = SimpleNamespace(forward_batch_generation=target_forward)
    worker._make_next_draft_input_decode = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(
        worker_module, "eagle_prepare_for_verify", lambda *args: (forward, graph_replay)
    )
    published = []
    result = worker.forward_batch_generation(
        batch, on_publish=lambda lens: published.append(lens.clone())
    )
    assert epilogue.calls == 1
    assert result.accept_lens.tolist() == [1]
    assert result.next_token_ids[0].item() == 11
    assert published[0].tolist() == [4]
    assert result.extra_keep_alive_refs == [forward]
    epilogue.predict_buf.zero_()
    epilogue.compact_predict_buf.zero_()
    epilogue.accept_lens_buf.zero_()
    assert result.accept_lens.tolist() == [1]
    assert result.next_token_ids[0].item() == 11


def test_worker_rejects_penalties_before_prefill():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.ubtree_worker_v2 import UBTreeWorkerV2

    worker = UBTreeWorkerV2.__new__(UBTreeWorkerV2)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=True, logit_bias={1: 1}),
    )
    with pytest.raises(ValueError, match="without penalties"):
        worker.forward_batch_generation(batch)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
