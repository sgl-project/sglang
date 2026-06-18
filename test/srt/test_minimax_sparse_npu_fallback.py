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


def test_npu_block_topk_counts_init_and_local_inside_topk_budget():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.block_size_k = 2
    backend.topk_blocks = 2
    backend.init_blocks = 1
    backend.local_blocks = 1
    backend.score_type = "max"

    scores = torch.tensor([[0.0, 0.0, 5.0, 5.0, 9.0, 9.0, 1.0, 1.0]])

    topk_idx = backend._block_topk(scores, seq_len=8)

    assert topk_idx.shape == (1, 2)
    torch.testing.assert_close(topk_idx, torch.tensor([[0, 3]], dtype=torch.int32))


def test_npu_cache_slot_gather_preserves_3d_shape_and_values():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.is_npu = False

    slots = torch.arange(5 * 2 * 3).reshape(5, 2, 3)
    locs = torch.tensor([3, 1], dtype=torch.int64)

    gathered = backend._gather_cache_slots(slots, locs, "unit test")

    assert gathered.shape == (2, 2, 3)
    torch.testing.assert_close(gathered, slots[locs])


def test_npu_cache_slot_gather_supports_torch_gather_path():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.is_npu = False

    slots = torch.arange(5 * 1 * 4).reshape(5, 1, 4)
    locs = torch.tensor([4, 2, 0], dtype=torch.int32)

    gathered = backend._gather_cache_slots(
        slots, locs, "unit test", use_torch_gather=True
    )

    assert gathered.shape == (3, 1, 4)
    torch.testing.assert_close(gathered, slots[locs.long()])
