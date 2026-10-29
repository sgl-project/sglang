# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See python/sglang/srt/speculative/ubtree_components/LICENSE.

"""UBTree proposal, verify metadata, and cache-commit CUDA Graph regressions."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config
from sglang.srt.speculative.ubtree_components.epilogue import UBTreeVerifyEpilogue
from sglang.srt.speculative.ubtree_components.selector import UBTreeSelector
from sglang.srt.speculative.ubtree_components.tree_graph import build_ubtree_proposal
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


def _make_fa3_verify_backend(width, algorithm, enabled, *, max_bs=1):
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    from sglang.srt.runtime_context import get_context, get_parallel

    pool = SimpleNamespace(
        req_to_token=torch.arange(3 * 512, device="cuda").reshape(3, 512)
    )
    kv_pool = SimpleNamespace()
    runner = SimpleNamespace(
        sliding_window_size=None,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            context_len=512,
            attention_arch=AttentionArch.MHA,
            is_local_attention_model=False,
            head_dim=64,
            hf_text_config=SimpleNamespace(num_attention_heads=4),
            get_num_kv_heads=lambda _: 2,
        ),
        device="cuda",
        req_to_token_pool=pool,
        token_to_kv_pool=kv_pool,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_dtype_str="bfloat16",
        page_size=1,
        is_draft_worker=False,
        kv_index_translator=KVIndexTranslator(
            req_to_token=pool.req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=kv_pool,
            page_size=1,
            device="cuda",
        ),
    )
    with (
        get_context().override_server_args(
            speculative_algorithm=algorithm,
            speculative_num_draft_tokens=width,
            speculative_eagle_topk=2,
        ),
        get_parallel().override(attn_tp_size=1),
        envs.SGLANG_OPT_FA3_VERIFY_METADATA_IN_GRAPH.override(enabled),
    ):
        backend = FlashAttentionBackend(runner)
        backend.init_cuda_graph_state(max_bs, max_bs * width)
    return backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 requires CUDA")
@pytest.mark.parametrize("width", [3, 17])
@pytest.mark.parametrize(
    "mode,batch_size,live_bs",
    [
        ("eager", 1, 1),
        ("eager", 2, 2),
        ("replay", 1, 1),
        ("replay", 2, 2),
        ("replay", 2, 1),
    ],
)
def test_eagle_full_mask_metadata_reuses_tree_columns(width, mode, batch_size, live_bs):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.eagle_utils import TreeMaskMode

    backend = _make_fa3_verify_backend(width, "EAGLE", True, max_bs=batch_size)
    lengths = [4, 7][:live_bs] + [0] * (batch_size - live_bs)
    visible = torch.ones(width, width, dtype=torch.bool, device="cuda").tril()
    masks = [
        torch.cat(
            (torch.ones(width, length, dtype=torch.bool, device="cuda"), visible), dim=1
        )
        for length in lengths[:live_bs]
    ]
    spec = SimpleNamespace(
        tree_mask_mode=TreeMaskMode.FULL_MASK,
        custom_mask=torch.cat([mask.flatten() for mask in masks]),
        positions=torch.arange(live_bs * width, device="cuda"),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=batch_size,
        req_pool_indices=torch.arange(batch_size, device="cuda"),
        seq_lens=torch.tensor(lengths, device="cuda"),
        seq_lens_cpu=torch.tensor(lengths),
        encoder_lens=None,
        spec_info=spec,
        positions=spec.positions,
        out_cache_loc=None,
        kv_loc_plan=None,
    )
    if mode == "replay":
        backend.init_forward_metadata_out_graph(batch, in_capture=True)

    with patch("torch.arange", wraps=torch.arange) as arange:
        if mode == "eager":
            backend.init_forward_metadata(batch)
        else:
            backend.init_forward_metadata_out_graph(batch)
    # Prefix cu-seqlens have different arange arguments. The tree-column
    # range must be constructed once and shared by mask/page-table gathers.
    tree_ranges = [call for call in arange.call_args_list if call.args == (width,)]
    assert len(tree_ranges) == 1
    metadata = backend.forward_metadata_spec_decode_expand
    torch.testing.assert_close(
        metadata.cache_seqlens_int32[: live_bs * width],
        visible.sum(1).int().repeat(live_bs),
    )
    columns = torch.arange(width, device="cuda").expand(width, -1)
    order = torch.where(visible, columns, columns + width).argsort(dim=1)
    expected = torch.cat(
        [
            backend.req_to_token[row, length : length + width]
            .expand(width, -1)
            .gather(1, order)
            for row, length in enumerate(lengths[:live_bs])
        ]
    )
    torch.testing.assert_close(
        metadata.page_table[: live_bs * width, :width].long(), expected.long()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 requires CUDA")
@pytest.mark.parametrize("width", [2, 17, 32, 33, 49, 65])
@pytest.mark.parametrize(
    "algorithm,enabled", [("UBTREE", True), ("UBTREE", False), ("EAGLE", True)]
)
def test_fa3_verify_metadata_graph_isolated_and_refreshes_live_inputs(
    width, algorithm, enabled
):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.eagle_utils import TreeMaskMode

    backend = _make_fa3_verify_backend(width, algorithm, enabled)
    optimized = algorithm == "UBTREE" and enabled
    assert backend._ubtree_verify_metadata_in_graph == optimized
    q = torch.randn(width, 4, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(3 * 512, 1, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    mode = TreeMaskMode.QLEN_ONLY if algorithm == "UBTREE" else TreeMaskMode.FULL_MASK
    spec = SimpleNamespace(
        tree_mask_mode=mode, positions=torch.arange(width, device="cuda")
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=1,
        req_pool_indices=torch.tensor([1], device="cuda"),
        seq_lens=torch.tensor([4], device="cuda"),
        seq_lens_cpu=None,
        encoder_lens=None,
        spec_info=spec,
        positions=spec.positions,
        out_cache_loc=None,
        kv_loc_plan=None,
    )

    def stage_mask(visible, length):
        if mode == TreeMaskMode.FULL_MASK:
            visible = torch.cat(
                (torch.ones(width, length, dtype=torch.bool, device="cuda"), visible),
                dim=1,
            )
        spec.custom_mask = visible.flatten().clone()

    visible = torch.ones(width, width, dtype=torch.bool, device="cuda").tril()
    stage_mask(visible, 4)
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    backend.init_forward_metadata_out_graph(batch)

    def attend(use_precomputed):
        results = []
        for metadata in (
            backend.forward_metadata,
            backend.forward_metadata_spec_decode_expand,
        ):
            result = backend.flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=metadata.max_seq_len_q,
                softmax_scale=64**-0.5,
                causal=False,
                return_softmax_lse=True,
                num_splits=backend.num_splits,
                scheduler_metadata=metadata.scheduler_metadata
                if use_precomputed
                else None,
            )
            results.extend(result[:2])
        return results

    def run():
        backend.init_forward_metadata_in_graph(batch)
        return attend(optimized)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    prefix = backend.forward_metadata
    suffix = backend.forward_metadata_spec_decode_expand
    metadata_pointers = (prefix.page_table.data_ptr(), suffix.page_table.data_ptr())
    scheduler_pointers = tuple(
        m.scheduler_metadata.data_ptr() if optimized else None for m in (prefix, suffix)
    )
    for iteration, (row, length) in enumerate(((2, 13), (1, 2), (2, 193))):
        batch.req_pool_indices.fill_(row)
        batch.seq_lens.fill_(length)
        q.copy_(torch.randn_like(q))
        if iteration == 1:
            backend.req_to_token[1].copy_(backend.req_to_token[1].flip(0))
        visible = torch.eye(width, dtype=torch.bool, device="cuda")
        visible[:, 0] = True
        if iteration == 2:
            visible = torch.ones_like(visible).tril()
        stage_mask(visible, length)
        backend.init_forward_metadata_out_graph(batch)
        graph.replay()
        torch.cuda.synchronize()
        assert metadata_pointers == (
            prefix.page_table.data_ptr(),
            suffix.page_table.data_ptr(),
        )
        assert prefix.cache_seqlens_int32.item() == length
        torch.testing.assert_close(
            prefix.page_table[0, :length].long(), backend.req_to_token[row, :length]
        )
        torch.testing.assert_close(suffix.cache_seqlens_int32, visible.sum(1).int())
        columns = torch.arange(width, device="cuda").expand(width, -1)
        order = torch.where(visible, columns, columns + width).argsort(dim=1)
        slots = backend.req_to_token[row, length : length + width].expand(width, -1)
        torch.testing.assert_close(suffix.page_table.long(), slots.gather(1, order))
        for i, metadata in enumerate((prefix, suffix)):
            if optimized:
                assert metadata.scheduler_metadata.data_ptr() == scheduler_pointers[i]
            else:
                assert metadata.scheduler_metadata is None
        # Compare both FA3 cascade legs with the ordinary per-call scheduler.
        expected = attend(False)
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_full_draft_graph_refreshes_replaced_request_tensors(monkeypatch):
    from sglang.kernels.ops.speculative.ubtree import prepare_draft_commit_metadata
    from sglang.srt.speculative.ubtree_components import draft_graph

    # No collectives in this small backbone. Keep the real CUDA capture/replay
    # and production runner, replacing only the distributed stream context.
    @contextmanager
    def capture_context(stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)

    monkeypatch.setattr(draft_graph, "graph_capture", capture_context)

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden, batch, residual):
            values = positions + batch.seq_lens[0] + batch.req_pool_indices[0]
            return hidden + values[:, None], None

    embedding = torch.nn.Embedding(17, 8, device="cuda")
    model = SimpleNamespace(layers=[Layer()], norm=torch.nn.Identity())
    backend = SimpleNamespace(
        init_forward_metadata_in_graph=lambda *args: None,
        init_forward_metadata_out_graph=lambda *args, **kwargs: None,
    )
    runner = draft_graph.UBTreeDraftSegmentGraphRunner(
        model=model,
        input_embedding=embedding,
        attn_backend=backend,
        batch_sizes=[1],
        block_size=4,
        hidden_size=8,
        device="cuda",
    )
    req_to_token = torch.arange(4 * 64, device="cuda").view(4, 64)
    tail_positions = torch.empty((1, 4), dtype=torch.int64, device="cuda")
    tail_cache_locs = torch.empty_like(tail_positions)

    def tail(output, input_ids, batch):
        prepare_draft_commit_metadata(
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token,
            positions_out=tail_positions,
            cache_loc_out=tail_cache_locs,
        )
        return tail_positions, tail_cache_locs

    runner.attach_capture_tail(tail)
    batches = []
    for seq_len, req_index in ((5, 1), (11, 2), (3, 1)):
        # Retain the previous tensors so new inputs cannot reuse their addresses.
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.tensor(
                [req_index], dtype=torch.int32, device="cuda"
            ),
            out_cache_loc=torch.arange(4, device="cuda") + seq_len,
            mark_forward_metadata_ready=lambda: None,
        )
        batches.append(batch)
        tokens = torch.arange(4, device="cuda")
        positions = torch.arange(4, device="cuda") + seq_len
        actual = runner(input_ids=tokens, positions=positions, forward_batch=batch)
        torch.cuda.synchronize()
        expected = embedding(tokens) + (positions + seq_len + req_index)[:, None]
        torch.testing.assert_close(actual.hidden_states, expected, rtol=0, atol=0)
        torch.testing.assert_close(tail_positions, positions[None, :])
        torch.testing.assert_close(
            tail_cache_locs, req_to_token[req_index, positions][None, :]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_draft_commit_metadata_is_cuda_graph_replay_safe():
    from sglang.kernels.ops.speculative.ubtree import prepare_draft_commit_metadata

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
    from sglang.kernels.ops.speculative.ubtree import prepare_target_kv_move_metadata

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
    from sglang.kernels.ops.speculative.ubtree import prepare_unified_commit_metadata

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_sampling_graph_refreshes_coins_parameters_and_logits(monkeypatch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.speculative.ubtree_components import sampling_graph
    from sglang.srt.speculative.ubtree_components.sampling_graph import (
        UBTreeSamplingGraphRunner,
    )

    @contextmanager
    def capture_context(stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)

    # Keep real capture/replay; this one-device fixture has no TP process group.
    monkeypatch.setattr(sampling_graph, "graph_capture", capture_context)

    width, vocab_size = 33, 64
    epilogue = UBTreeVerifyEpilogue(
        max_bs=1, node_count=width, max_depth=2, device="cuda"
    )
    metadata = SimpleNamespace(
        batch_size=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
        seq_lens=torch.tensor([7], dtype=torch.int64, device="cuda"),
        out_cache_loc=torch.arange(width, device="cuda"),
        mamba_track_indices=None,
    )
    tree = SimpleNamespace(
        batch_size=1,
        tokens=torch.arange(width, device="cuda").view(1, -1),
        retrieve_index=torch.arange(width, device="cuda").view(1, -1),
        retrieve_next_token=torch.full((1, width), -1, device="cuda"),
        retrieve_next_sibling=torch.full((1, width), -1, device="cuda"),
    )
    tree.retrieve_next_token[0, 0] = 1
    tree.retrieve_next_sibling[0, 1:-1] = torch.arange(2, width, device="cuda")
    epilogue.begin_tree_step(tree, metadata.seq_lens)
    logits = torch.zeros((width, vocab_size), device="cuda")
    hidden = torch.randn((width, 8), device="cuda")
    epilogue.capture_hook(
        SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False)),
        LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden),
        metadata,
        width,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        top_ks=torch.tensor([20], dtype=torch.int32, device="cuda"),
        top_ps=torch.ones((1,), device="cuda"),
        min_ps=torch.zeros((1,), device="cuda"),
        need_min_p_sampling=False,
        is_all_greedy=False,
        sampling_seed=None,
    )
    runner = UBTreeSamplingGraphRunner("cuda")
    assert runner.device == logits.device
    runner._generator.manual_seed(314159)

    def run(graph=True):
        return runner.run(
            logits=logits,
            hidden=hidden,
            sampling_info=sampling_info,
            epilogue=epilogue,
            forward_batch=metadata,
            can_run_cuda_graph=graph,
        )

    with patch(
        "sglang.srt.speculative.ubtree_components.epilogue.get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    ):
        samples = []
        for _ in range(4):
            run()
            captured = next(iter(runner._captured.values()))
            samples.append(captured.target_predict.clone())
        assert all(not torch.equal(samples[0], sample) for sample in samples[1:])
        # A full-vocabulary top-k cutoff retains all equal-probability ties.
        assert torch.cat(samples).max().item() >= 20
        assert len(runner._captured) == 1

        logits.fill_(-100)
        logits[:, 3] = 0
        logits[:, 5] = -0.1
        sampling_info.top_ps.fill_(0.01)
        run()
        assert torch.all(captured.target_predict == 3)
        assert len(runner._captured) == 1
        logits[:, 3] = -100
        logits[:, 5] = 0
        run()
        assert torch.all(captured.target_predict == 5)

        sampling_info.need_min_p_sampling = True
        sampling_info.min_ps.fill_(0.9)
        sampling_info.top_ps.fill_(1.0)
        run()
        assert len(runner._captured) == 2
        assert epilogue.read_result(1).predict[0].item() == 5
        result = run(graph=False)
        assert result.predict[0].item() == 5
        assert result.accept_lens.item() == 2
        assert result.new_seq_lens.item() == 9
        # Eager metadata may alias the previous relay output.
        metadata.seq_lens = result.new_seq_lens
        epilogue.begin_tree_step(tree, metadata.seq_lens)
        result = run(graph=False)
        assert result.new_seq_lens.item() == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
@pytest.mark.parametrize("capture_sampling", [False, True])
def test_verify_graphs_retain_materializer_workspace_after_prefill_growth(
    monkeypatch, capture_sampling
):
    from sglang.kernels.ops.speculative.fused_kv_materialize import (
        FusedKVMaterializeHelper,
    )
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.ubtree_components import sampling_graph

    @contextmanager
    def capture_context(stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)

    monkeypatch.setattr(sampling_graph, "graph_capture", capture_context)
    rotary = SimpleNamespace(
        rotary_dim=16,
        is_neox_style=True,
        cos_sin_cache=torch.cat(
            (torch.ones(512, 8, device="cuda"), torch.zeros(512, 8, device="cuda")),
            dim=1,
        ),
    )
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                num_kv_heads=1,
                head_dim=16,
                rotary_emb=rotary,
                q_size=16,
                kv_size=16,
                qkv_proj=SimpleNamespace(
                    weight=torch.randn(48, 8, device="cuda", dtype=torch.bfloat16)
                ),
                k_norm=SimpleNamespace(
                    weight=torch.ones(16, device="cuda", dtype=torch.bfloat16),
                    variance_epsilon=1e-6,
                ),
            )
        )
        for _ in range(2)
    ]
    helper = FusedKVMaterializeHelper(
        layers, rotary, 1, 16, torch.device("cuda"), max_position_hint=512
    )
    assert helper.get_workspace_tensors() == ()
    output_k = torch.empty(2, 2, 1, 16, device="cuda", dtype=torch.bfloat16)
    output_v = torch.empty_like(output_k)

    def write_kv(layer, k, v):
        output_k[layer].copy_(k)
        output_v[layer].copy_(v)

    def draft_commit(*, target_hidden, positions, **kwargs):
        helper.materialize(target_hidden, positions, write_kv)

    def grow_prefill_workspace(length):
        helper.materialize(
            torch.randn(length, 8, device="cuda", dtype=torch.bfloat16),
            torch.arange(length, device="cuda"),
            lambda layer, k, v: None,
        )
        torch.cuda.synchronize()
        gc.collect()

    def workspace_refs():
        return tuple(weakref.ref(tensor) for tensor in helper.get_workspace_tensors())

    epilogue = UBTreeVerifyEpilogue(max_bs=1, node_count=3, max_depth=2, device="cuda")
    epilogue.attach_draft_kv_commit(
        draft_commit=draft_commit,
        req_to_token=torch.arange(512, device="cuda").view(1, -1),
        get_workspace_tensors=helper.get_workspace_tensors,
    )
    metadata = SimpleNamespace(
        batch_size=1,
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens=torch.tensor([7], device="cuda"),
        out_cache_loc=torch.arange(3, device="cuda"),
        mamba_track_indices=None,
    )
    tree = SimpleNamespace(
        batch_size=1,
        tokens=torch.tensor([[0, 2, 3]], device="cuda"),
        retrieve_index=torch.arange(3, device="cuda").view(1, -1),
        retrieve_next_token=torch.tensor([[1, 2, -1]], device="cuda"),
        retrieve_next_sibling=torch.full((1, 3), -1, device="cuda"),
    )
    epilogue.begin_tree_step(tree, metadata.seq_lens)
    logits = torch.full((3, 32), -100.0, device="cuda")
    logits[:, 2] = 0
    hidden = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
    output = LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)

    def target_tail():
        epilogue.capture_hook(
            SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False)),
            output,
            metadata,
            3,
        )

    target_tail()
    target_refs = workspace_refs()
    assert len(target_refs) == 3
    assert epilogue._graph_draft_kv_workspaces == {}
    target_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(target_graph):
        target_tail()
    target_graph.replay()
    expected_k, expected_v = output_k.clone(), output_v.clone()

    # Prefill replaces all three workspaces captured by the folded target tail.
    grow_prefill_workspace(80)
    assert all(ref() is not None for ref in target_refs)
    sampling_refs = workspace_refs()
    runner = sampling_graph.UBTreeSamplingGraphRunner("cuda")
    sampling_info = SimpleNamespace(
        temperatures=torch.ones(1, 1, device="cuda"),
        top_ks=torch.tensor([20], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.95], device="cuda"),
        min_ps=torch.zeros(1, device="cuda"),
        need_min_p_sampling=False,
        is_all_greedy=False,
        sampling_seed=None,
    )

    def sampling_tail():
        return runner.run(
            logits=logits,
            hidden=hidden,
            sampling_info=sampling_info,
            epilogue=epilogue,
            forward_batch=metadata,
            can_run_cuda_graph=True,
        )

    with get_parallel().override(tp_group=SimpleNamespace(world_size=1)):
        if capture_sampling:
            sampling_tail()
        # A second growth must also preserve the separate sampling tail's
        # workspace. Uncaptured intermediate prefill buffers remain reclaimable.
        grow_prefill_workspace(200)
        uncaptured_refs = workspace_refs()
        grow_prefill_workspace(400)
        assert all(ref() is None for ref in uncaptured_refs)
        assert all(ref() is not None for ref in target_refs)
        assert all((ref() is not None) == capture_sampling for ref in sampling_refs)
        torch.cuda.empty_cache()
        epilogue.set_armed(True)
        target_graph.replay()
        torch.testing.assert_close(output_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(output_v, expected_v, rtol=0, atol=0)
        if capture_sampling:
            assert sampling_tail().accept_lens.item() == 2
            torch.testing.assert_close(output_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(output_v, expected_v, rtol=0, atol=0)
            runner._captured.clear()
            gc.collect()
            assert all(ref() is None for ref in sampling_refs)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
