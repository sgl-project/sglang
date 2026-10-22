# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See python/sglang/srt/speculative/ubtree_components/LICENSE.

"""UBTree proposal, verify metadata, and cache-commit CUDA Graph regressions."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.ubtree_components.epilogue import UBTreeVerifyEpilogue
from sglang.srt.speculative.ubtree_components.selector import UBTreeSelector
from sglang.srt.speculative.ubtree_components.tree import build_sglang_full_mask
from sglang.srt.speculative.ubtree_components.tree_graph import (
    UBTreeProposalGraphRunner,
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


def _make_proposal_graph_runner(device, *, batch_sizes):
    config, selector, lm_head = _make_selector_graph_components(device)
    return UBTreeProposalGraphRunner(
        batch_sizes=batch_sizes,
        depth_count=4,
        hidden_size=config.hidden_size,
        hidden_dtype=torch.float32,
        candidate_k=3,
        vocab_size=config.vocab_size,
        lm_head=lm_head,
        selector=selector,
        budget=5,
        supertree_width=3,
        depth_bonus=0.0,
        device=device,
    )


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
def test_proposal_graph_runner_replays_changing_inputs_exactly():
    runner = _make_proposal_graph_runner("cuda", batch_sizes=[2])
    eager = UBTreeProposalGraphRunner(
        batch_sizes=[],
        depth_count=runner.depth_count,
        hidden_size=runner.hidden_size,
        hidden_dtype=runner.hidden_dtype,
        candidate_k=runner.candidate_k,
        vocab_size=runner.vocab_size,
        lm_head=runner.lm_head,
        selector=runner.selector,
        budget=runner.budget,
        supertree_width=runner.supertree_width,
        depth_bonus=runner.depth_bonus,
        device="cuda",
    )
    for seed in (61, 62):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        hidden = torch.randn(2, 4, 8, device="cuda", generator=generator)
        bonus_tokens = torch.randint(0, 17, (2,), device="cuda", generator=generator)
        expected = eager(hidden=hidden, bonus_tokens=bonus_tokens)
        actual = runner(hidden=hidden, bonus_tokens=bonus_tokens)
        torch.cuda.synchronize()
        _assert_tree_batches_equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton mask builder requires CUDA"
)
def test_verify_metadata_cuda_mask_builder_is_exact_and_stable():
    from sglang.srt.speculative.ubtree_components.verify_metadata import (
        UBTreeVerifyMetadataBuffers,
    )

    metadata = UBTreeVerifyMetadataBuffers(
        batch_sizes=(2, 4),
        num_nodes=5,
        max_context_len=32,
        device="cuda",
    )
    pointer = metadata.custom_mask.data_ptr()
    for seed, prefix_lens in ((71, [3, 7]), (72, [9]), (73, [1, 4, 8, 11])):
        raw_bs = len(prefix_lens)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        visibility = torch.rand(raw_bs, 5, 5, generator=generator, device="cuda") > 0.5
        visibility.diagonal(dim1=1, dim2=2).fill_(True)
        prefix_lens_tensor = torch.tensor(prefix_lens, device="cuda")
        actual = metadata.prepare_custom_mask(
            visibility,
            prefix_lens_tensor,
            prefix_lens_sum=sum(prefix_lens),
        )
        expected = build_sglang_full_mask(
            visibility,
            prefix_lens_tensor,
            prefix_lens_sum=sum(prefix_lens),
        )
        padded_bs = metadata.graph_batch_size(raw_bs)
        if padded_bs > raw_bs:
            expected = torch.cat(
                (
                    expected,
                    torch.ones(
                        (padded_bs - raw_bs) * 5 * (1 + 5),
                        dtype=torch.bool,
                        device="cuda",
                    ),
                )
            )
        torch.cuda.synchronize()
        assert actual.data_ptr() == pointer
        assert torch.equal(actual, expected)

        packed = torch.empty(
            metadata.custom_mask.numel(), dtype=torch.uint8, device="cuda"
        )
        packed_indptr = torch.empty(
            metadata.max_bs + 1, dtype=torch.int32, device="cuda"
        )
        packed_pointer = packed.data_ptr()
        indptr_pointer = packed_indptr.data_ptr()
        assert metadata.prepare_packed_custom_mask(
            packed, packed_indptr, batch_size=padded_bs
        )

        padded_prefix_lens = torch.ones(padded_bs, dtype=torch.int64, device="cuda")
        padded_prefix_lens[:raw_bs].copy_(prefix_lens_tensor)
        bit_indptr = torch.zeros(padded_bs + 1, dtype=torch.int64, device="cuda")
        bit_indptr[1:] = torch.cumsum(5 * (padded_prefix_lens + 5), dim=0)
        from flashinfer.quantization import segment_packbits

        expected_packed, expected_packed_indptr = segment_packbits(
            actual, bit_indptr, bitorder="little"
        )
        torch.cuda.synchronize()
        assert packed.data_ptr() == packed_pointer
        assert packed_indptr.data_ptr() == indptr_pointer
        assert torch.equal(packed_indptr[: padded_bs + 1], expected_packed_indptr)
        assert torch.equal(packed[: expected_packed.numel()], expected_packed)


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
def test_target_kv_move_graph_replays_with_changing_accept_paths():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.ubtree_components.target_kv_graph import (
        UBTreeTargetKVMoveGraphRunner,
    )

    device = torch.device("cuda")
    pool = MHATokenToKVPool(
        size=255,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=2,
        device=str(device),
        enable_memory_saver=False,
        enable_alt_stream=False,
        enable_kv_cache_copy=True,
    )
    req_to_token = torch.zeros((4, 64), dtype=torch.int64, device=device)
    req_to_token[1, 7:11] = torch.tensor([101, 102, 103, 104], device=device)
    req_to_token[3, 13:17] = torch.tensor([121, 122, 123, 124], device=device)

    runner = UBTreeTargetKVMoveGraphRunner(
        kvcache=pool,
        req_to_token=req_to_token,
        batch_sizes=(2,),
        accept_width=4,
        verify_node_count=5,
        device=device,
    )
    assert runner.batch_sizes == (2,)

    for layer_idx, (key, value) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
        rows = torch.arange(key.shape[0], dtype=key.dtype, device=device)
        key.copy_(rows[:, None, None] + 1000 * layer_idx)
        value.copy_(rows[:, None, None] + 2000 * layer_idx)

    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([7, 13], dtype=torch.int32, device=device)
    accept_index = torch.tensor(
        [[1, 3, -1, -1], [5, 9, 7, -1]], dtype=torch.int32, device=device
    )
    accept_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    out_cache_loc = torch.tensor(
        [20, 21, 22, 23, 24, 30, 31, 32, 33, 34],
        dtype=torch.int64,
        device=device,
    )
    assert runner(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        accept_index=accept_index,
        accept_lens=accept_lens,
        out_cache_loc=out_cache_loc,
    )
    torch.cuda.synchronize()

    first_targets = torch.tensor([101, 102, 121, 122, 123], device=device)
    first_sources = torch.tensor([21, 23, 30, 34, 32], device=device)
    for layer_idx, (key, value) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
        assert torch.equal(
            key[first_targets],
            (first_sources.to(key.dtype)[:, None, None] + 1000 * layer_idx).expand_as(
                key[first_targets]
            ),
        )
        assert torch.equal(
            value[first_targets],
            (first_sources.to(value.dtype)[:, None, None] + 2000 * layer_idx).expand_as(
                value[first_targets]
            ),
        )

    # Replay the same graph with different lengths and path indices.
    seq_lens.copy_(torch.tensor([9, 14], dtype=torch.int32, device=device))
    accept_index.copy_(torch.tensor([[4, -1, -1, -1], [6, 8, -1, -1]], device=device))
    accept_lens.copy_(torch.tensor([1, 2], dtype=torch.int32, device=device))
    assert runner(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        accept_index=accept_index,
        accept_lens=accept_lens,
        out_cache_loc=out_cache_loc,
    )
    torch.cuda.synchronize()

    second_targets = torch.tensor([103, 122, 123], device=device)
    second_sources = torch.tensor([24, 31, 33], device=device)
    for layer_idx, (key, value) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
        assert torch.equal(
            key[second_targets],
            (second_sources.to(key.dtype)[:, None, None] + 1000 * layer_idx).expand_as(
                key[second_targets]
            ),
        )
        assert torch.equal(
            value[second_targets],
            (
                second_sources.to(value.dtype)[:, None, None] + 2000 * layer_idx
            ).expand_as(value[second_targets]),
        )


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


def test_verify_epilogue_folds_factorized_kda_commit(monkeypatch):
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
    track_indices = torch.tensor([21, 22], dtype=torch.int64)
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
    assert torch.equal(committed["last_correct_step_indices"], torch.tensor([2, 3]))
    assert torch.equal(committed["mamba_track_indices"], track_indices)
    assert torch.equal(committed["mamba_steps_to_track"], torch.tensor([0, -1]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
