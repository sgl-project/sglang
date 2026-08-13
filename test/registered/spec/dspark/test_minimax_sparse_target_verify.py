from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeKVPool:
    def set_fused_kv_index_buffer(self, *args, **kwargs):
        pass

    def get_kv_buffer(self, layer_id):
        return torch.empty(1), torch.empty(1)

    def get_index_kv_buffer(self, layer_id):
        return torch.empty(1), torch.empty(1)


def _make_backend():
    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.kv_pool = _FakeKVPool()
    backend.req_to_token = torch.empty(1, dtype=torch.int32)
    backend.disable_value_layer_ids = set()
    backend.fp8_attn_gemm = False
    backend._max_seqlen_q = 8
    backend._max_seqlen_k = 128
    backend.block_size_q = 64
    backend.block_size_k = 64
    backend.topk_blocks = 8
    backend.init_blocks = 1
    backend.local_blocks = 1
    backend.score_type = "indexer"
    backend.use_msa = False
    return backend


def _make_layer():
    return SimpleNamespace(
        layer_id=0,
        k_scale_float=None,
        v_scale_float=None,
        idx_k_scale_float=None,
        idx_v_scale_float=None,
        q_scale_float=None,
        idx_q_scale_float=None,
    )


def _make_batch(*, layout, extend_seq_lens=None, extend_seq_lens_cpu=None):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(
            draft_token_num=2,
            ragged_verify_layout=layout,
        ),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens=None,
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        minimax_m3_precached_sparse_layers=None,
    )


def _run_forward(backend, batch):
    q = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    idx_q = q + 100
    k = torch.empty(4, 1)
    v = torch.empty(4, 1)
    idx_k = torch.empty(4, 1)
    idx_v = torch.empty(4, 1)
    return backend.forward_extend(
        q,
        k,
        v,
        _make_layer(),
        batch,
        idx_q=idx_q,
        idx_k=idx_k,
        idx_v=idx_v,
    )


def test_forward_batch_builds_uniform_capture_layout():
    batch = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=2,
        input_ids=torch.arange(4),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        out_cache_loc=torch.arange(4, dtype=torch.int32),
        seq_lens_sum=12,
        spec_info=SimpleNamespace(draft_token_num=2),
    )

    layout = batch.build_uniform_target_verify_layout(graph_num_tokens=4)

    assert layout.verify_lens.tolist() == [2, 2]
    assert layout.qo_indptr_device.tolist() == [0, 2, 4]
    assert layout.verify_lens_cpu == [2, 2]
    assert layout.total_verify_tokens == 4


def test_dp_padded_target_verify_requires_forward_batch_geometry():
    backend = _make_backend()
    batch = _make_batch(
        layout=None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
    )
    batch.num_token_non_padded_cpu = 3

    with patch(
        "sglang.srt.layers.attention.minimax_sparse_backend.minimax_sparse_prefill",
        return_value=(None, torch.empty(4, 2)),
    ):
        with pytest.raises(
            ValueError,
            match="requires ForwardBatch extend geometry",
        ):
            _run_forward(backend, batch)


def test_ragged_target_verify_trims_dp_padding_before_sparse_kernel():
    layout = RaggedVerifyLayout(
        verify_lens=torch.tensor([2, 1], dtype=torch.int32),
        graph_num_tokens=4,
        extend_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        qo_indptr_device=torch.tensor([0, 2, 3], dtype=torch.int32),
        verify_lens_cpu=[2, 1],
        total_verify_tokens=3,
    )
    backend = _make_backend()
    batch = _make_batch(
        layout=layout,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
    )
    captured = {}

    def fake_sparse_prefill(*args, **kwargs):
        captured["q"] = args[0].clone()
        captured["cu_seqlens_q"] = args[10].clone()
        captured["cache_seqlens"] = args[11].clone()
        captured["verify_lens_cpu"] = kwargs["seqlens_cpu"]
        return None, torch.ones(args[0].shape[0], 2)

    with patch(
        "sglang.srt.layers.attention.minimax_sparse_backend.minimax_sparse_prefill",
        side_effect=fake_sparse_prefill,
    ):
        _, output = _run_forward(backend, batch)

    assert captured["q"].shape[0] == 3
    assert captured["cu_seqlens_q"].tolist() == [0, 2, 3]
    assert captured["cache_seqlens"].tolist() == [7, 8]
    assert captured["verify_lens_cpu"] == [2, 1]
    assert output.shape == (4, 2)
    assert output[-1].tolist() == [0.0, 0.0]


def test_forward_batch_target_verify_geometry_trims_dp_padding():
    backend = _make_backend()
    batch = _make_batch(
        layout=None,
        extend_seq_lens=torch.tensor([2, 0], dtype=torch.int32),
        extend_seq_lens_cpu=[2, 0],
    )
    batch.extend_prefix_lens = batch.seq_lens
    captured = {}

    def fake_sparse_prefill(*args, **kwargs):
        captured["q"] = args[0].clone()
        captured["cu_seqlens_q"] = args[10].clone()
        captured["cache_seqlens"] = args[11].clone()
        captured["prefix_lens"] = args[12].clone()
        captured["verify_lens_cpu"] = kwargs["seqlens_cpu"]
        return None, torch.ones(args[0].shape[0], 2)

    with patch(
        "sglang.srt.layers.attention.minimax_sparse_backend.minimax_sparse_prefill",
        side_effect=fake_sparse_prefill,
    ):
        _, output = _run_forward(backend, batch)

    assert captured["q"].shape[0] == 2
    assert captured["cu_seqlens_q"].tolist() == [0, 2, 2]
    assert captured["cache_seqlens"].tolist() == [7, 7]
    assert captured["prefix_lens"].tolist() == [5, 7]
    assert captured["verify_lens_cpu"] == [2, 0]
    assert output.shape == (4, 2)
    assert output[2:].tolist() == [[0.0, 0.0], [0.0, 0.0]]
