import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _install_fake_modules():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.configs.model_config",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.base_attn_backend",
        "sglang.srt.layers.attention.minimax_sparse_ops",
        "sglang.srt.layers.attention.minimax_sparse_ops.common",
        "sglang.srt.layers.attention.minimax_sparse_ops.common.index",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.utils",
        "sglang.srt.utils.async_probe",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    model_config = sys.modules["sglang.srt.configs.model_config"]
    model_config.get_minimax_sparse_attention_config = lambda _cfg: {}
    model_config.get_minimax_sparse_disable_value_layer_ids = lambda _cfg: []
    model_config.get_minimax_sparse_layer_ids = lambda _cfg: ([], [])
    model_config.get_minimax_sparse_score_type = lambda _cfg: "max"

    base_attn = sys.modules["sglang.srt.layers.attention.base_attn_backend"]
    base_attn.AttentionBackend = type("AttentionBackend", (), {})

    common_index = sys.modules[
        "sglang.srt.layers.attention.minimax_sparse_ops.common.index"
    ]
    common_index.topk_index_reduce = lambda tensor, dim: tensor

    memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
    memory_pool.MiniMaxSparseKVPool = type("MiniMaxSparseKVPool", (), {})

    forward_batch = sys.modules["sglang.srt.model_executor.forward_batch_info"]
    forward_batch.ForwardBatch = type("ForwardBatch", (), {})

    utils = sys.modules["sglang.srt.utils"]
    utils.get_bool_env_var = lambda _name, default: default == "True"
    utils.is_npu = lambda: True

    async_probe = sys.modules["sglang.srt.utils.async_probe"]
    async_probe.maybe_detect_oob = lambda *args, **kwargs: None


def _load_minimax_sparse_backend_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_minimax_sparse_backend_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_npu_sparse_block_selection_masks_future_blocks_and_dedups_local():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.block_size_k = 2
    backend.topk_blocks = 1
    backend.init_blocks = 0
    backend.local_blocks = 1
    backend.score_type = "max"

    idx_q = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    idx_k = torch.tensor([[0.0], [1.0], [100.0], [100.0], [200.0], [200.0]])
    query_positions = torch.tensor([0, 1, 2], dtype=torch.long)

    blocks = backend._select_sparse_blocks(idx_q, idx_k, query_positions, seq_len=6)

    assert blocks.shape == (3, 1, 2)
    expected = torch.tensor([[[0, -1]], [[0, -1]], [[1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(blocks, expected)


def test_npu_sparse_seq_matches_dense_attention_when_all_blocks_are_selected():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.block_size_k = 2
    backend.topk_blocks = 4
    backend.init_blocks = 0
    backend.local_blocks = 0
    backend.score_type = "max"

    q_seq = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.bfloat16)
    k_seq = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]],
        dtype=torch.bfloat16,
    )
    v_seq = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
        dtype=torch.bfloat16,
    )
    idx_q = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    idx_k = torch.ones((4, 1), dtype=torch.bfloat16)
    query_positions = torch.tensor([0, 1], dtype=torch.long)

    _, out = backend._npu_sparse_seq(
        q_seq, k_seq, v_seq, idx_q, idx_k, None, query_positions, seq_len=4
    )

    scores = torch.einsum("qhd,khd->qhk", q_seq.float(), k_seq.float()) * (
        q_seq.shape[-1] ** -0.5
    )
    key_pos = torch.arange(k_seq.shape[0])
    valid = key_pos[None, :] <= query_positions[:, None]
    scores = scores.masked_fill(~valid[:, None, :], -1.0e30)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs.to(v_seq.dtype), v_seq)

    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-3, atol=1e-3)


def test_npu_triton_forward_metadata_reuses_block_table():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.page_size = 2
    backend.block_size_q = 1
    backend.block_size_k = 2
    backend._max_seqlen_q = 2
    backend._max_seqlen_k = 4
    backend.req_to_token = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.int32,
    )
    backend._npu_triton_forward_meta = None

    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
        seq_lens=torch.tensor([4, 3], dtype=torch.int32),
        extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([2, 2], dtype=torch.int32),
        extend_seq_lens_cpu=[2, 1],
    )

    backend._prepare_npu_triton_forward_meta(forward_batch)

    meta = backend._npu_triton_forward_meta
    assert meta is not None
    torch.testing.assert_close(
        meta.block_table,
        torch.tensor([[2, 3], [0, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        meta.cu_seqlens,
        torch.tensor([0, 2, 3], dtype=torch.int32),
    )
    assert meta.actual_num_tokens == 3


def test_npu_forward_extend_has_triton_prefill_gate():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
    ).read_text()

    assert "def _forward_npu_triton_prefill(" in source
    assert "if _npu_use_triton_sparse():" in source
    assert "self._forward_npu_triton_prefill(" in source


def test_minimax_sparse_triton_allocator_calls_are_guarded():
    ops_dir = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops"
    )
    offenders = []
    for path in ops_dir.rglob("*.py"):
        if "triton.set_allocator(" in path.read_text():
            offenders.append(str(path.relative_to(ops_dir)))

    assert offenders == []
