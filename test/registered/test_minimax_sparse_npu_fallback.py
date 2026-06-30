import importlib.util
import sys
import types
from pathlib import Path

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
        "triton",
        "triton.language",
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

    triton_mod = sys.modules["triton"]
    triton_mod.jit = lambda fn=None, **_kwargs: (lambda f: f) if fn is None else fn
    triton_mod.heuristics = lambda _values: (lambda fn: fn)
    triton_mod.next_power_of_2 = lambda x: 1 << (int(x) - 1).bit_length()
    triton_mod.language = sys.modules["triton.language"]

    tl_mod = sys.modules["triton.language"]
    tl_mod.constexpr = int


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
