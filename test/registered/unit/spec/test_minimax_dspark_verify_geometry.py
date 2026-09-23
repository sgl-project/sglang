from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM
from sglang.srt.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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
    backend.fp8_attn_gemm = backend.is_npu = backend.use_msa = False
    backend._max_seqlen_q, backend._max_seqlen_k = 8, 128
    backend.block_size_q = backend.block_size_k = 64
    backend.topk_blocks, backend.init_blocks, backend.local_blocks = 8, 1, 1
    backend.score_type = "indexer"
    backend._prefill_seqblock_meta = None
    backend.index_cache_enabled = False
    backend._loc_mapping = None
    return backend


def _make_batch(**overrides):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=2, ragged_verify_layout=None),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens=None,
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        minimax_m3_precached_sparse_layers=None,
        global_num_token_non_padded_cpu=None,
        attn_tp_sequence_sharded=False,
    )
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _run_forward(backend, batch):
    q = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    layer = SimpleNamespace(
        layer_id=0,
        k_scale_float=None,
        v_scale_float=None,
        idx_k_scale_float=None,
        idx_v_scale_float=None,
        q_scale_float=None,
        idx_q_scale_float=None,
    )
    return backend.forward_extend(
        q,
        torch.empty(4, 1),
        torch.empty(4, 1),
        layer,
        batch,
        idx_q=q + 100,
        idx_k=torch.empty(4, 1),
        idx_v=torch.empty(4, 1),
    )


@pytest.mark.parametrize(
    "global_num_tokens,sequence_sharded",
    [(2, False), (4, True)],
)
def test_non_ragged_dp_verify_requires_per_request_geometry(
    global_num_tokens, sequence_sharded
):
    batch = _make_batch(
        global_num_token_non_padded_cpu=global_num_tokens,
        attn_tp_sequence_sharded=sequence_sharded,
    )
    with patch(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
        "minimax_sparse_prefill",
        return_value=(None, torch.empty(4, 2)),
    ):
        with pytest.raises(RuntimeError, match="requires per-request geometry"):
            _run_forward(_make_backend(), batch)


def test_ragged_verify_trims_dp_padding_and_repads_output():
    layout = RaggedVerifyLayout(
        verify_lens=torch.tensor([2, 1], dtype=torch.int32),
        graph_num_tokens=4,
        extend_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        qo_indptr_device=torch.tensor([0, 2, 3], dtype=torch.int32),
        verify_lens_cpu=[2, 1],
        total_verify_tokens=3,
    )
    batch = _make_batch()
    batch.spec_info.ragged_verify_layout = layout
    captured = {}

    def fake_sparse_prefill(*args, **kwargs):
        captured["q_rows"] = args[0].shape[0]
        captured["cu_seqlens_q"] = args[10].tolist()
        captured["cache_seqlens"] = args[11].tolist()
        return None, torch.ones(args[0].shape[0], 2)

    with patch(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
        "minimax_sparse_prefill",
        side_effect=fake_sparse_prefill,
    ):
        _, output = _run_forward(_make_backend(), batch)

    assert captured == {
        "q_rows": 3,
        "cu_seqlens_q": [0, 2, 3],
        "cache_seqlens": [7, 8],
    }
    assert output.shape == (4, 2)
    assert output[-1].tolist() == [0.0, 0.0]


def test_forward_batch_geometry_trims_uniform_dp_padding():
    batch = _make_batch(
        extend_seq_lens=torch.tensor([2, 0], dtype=torch.int32),
        extend_seq_lens_cpu=[2, 0],
        extend_prefix_lens=torch.tensor([5, 7], dtype=torch.int32),
        seq_lens=torch.tensor([7, 7], dtype=torch.int32),
    )
    captured = {}

    def fake_sparse_prefill(*args, **kwargs):
        captured["q_rows"] = args[0].shape[0]
        captured["cu_seqlens_q"] = args[10].tolist()
        captured["verify_lens_cpu"] = kwargs["seqlens_cpu"]
        return None, torch.ones(args[0].shape[0], 2)

    with patch(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
        "minimax_sparse_prefill",
        side_effect=fake_sparse_prefill,
    ):
        _, output = _run_forward(_make_backend(), batch)

    assert captured == {
        "q_rows": 2,
        "cu_seqlens_q": [0, 2, 2],
        "verify_lens_cpu": [2, 0],
    }
    assert output[2:].tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize(
    "model_cls",
    [MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration],
)
def test_dspark_capture_reuses_eagle3_setup(model_cls):
    captured_layer_ids = []
    model = SimpleNamespace(
        set_eagle3_layers_to_capture=lambda ids: captured_layer_ids.append(ids)
    )

    model_cls.set_dspark_layers_to_capture(model, [0, 2])

    assert captured_layer_ids == [[0, 2]]
