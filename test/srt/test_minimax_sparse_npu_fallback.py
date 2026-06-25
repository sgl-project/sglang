import importlib.util
import os
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


def _load_npu_triton_prefill_module():
    _install_fake_modules()
    sys.modules.setdefault(
        "sglang.srt.layers.attention.minimax_sparse_ops.npu_triton",
        types.ModuleType("sglang.srt.layers.attention.minimax_sparse_ops.npu_triton"),
    )

    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_minimax_sparse_npu_prefill_under_test", module_path
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


def test_npu_prefill_debug_diff_location_formats_request_context():
    module = _load_minimax_sparse_backend_module()
    diff = torch.zeros((3, 2, 4), dtype=torch.float32)
    diff[2, 1, 3] = 2.0

    location = module._format_prefill_diff_location(
        diff,
        cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32),
        prefix_lens=torch.tensor([4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([5, 10], dtype=torch.int32),
        req_pool_indices=torch.tensor([7, 9], dtype=torch.int64),
        block_size_k=2,
    )

    assert location == (
        "token=2,head=1,dim=3,batch=1,req=9,q_offset=1,"
        "prefix_len=8,eff_seq_len=10,seq_len=10,block=4"
    )


def test_npu_prefill_triton_min_seqlen_defaults_to_zero_and_reads_env():
    module = _load_minimax_sparse_backend_module()
    env_name = module._NPU_PREFILL_TRITON_MIN_SEQLEN_ENV
    old_value = os.environ.get(env_name)
    try:
        os.environ.pop(env_name, None)
        assert module._npu_prefill_triton_min_seqlen() == 0

        os.environ[env_name] = "8192"
        assert module._npu_prefill_triton_min_seqlen() == 8192
    finally:
        if old_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_value


def test_npu_forward_extend_has_triton_prefill_gate():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
    ).read_text()

    assert "def _forward_npu_triton_prefill(" in source
    assert "if _npu_use_triton_sparse():" in source
    assert "self._forward_npu_triton_prefill(" in source


def test_npu_triton_prefill_uses_dedicated_prefill_kernels():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill.py"
    ).read_text()

    assert "flash_prefill_npu_with_topk_index" in source
    assert "flash_prefill_npu_with_gqa_share_sparse" in source
    assert "minimax_sparse_ops.prefill.flash_with_topk_idx" not in source
    assert "minimax_sparse_ops.prefill.topk_sparse" not in source
    assert "flash_decode_bnsd_with_topk_idx" not in source
    assert "flash_decode_bnsd_with_gqa_share_sparse" not in source


def test_npu_triton_prefill_appends_forced_blocks_after_pure_topk():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill.py"
    ).read_text()

    assert "_merge_prefill_sparse_blocks(" in source
    assert "init_blocks=0" in source
    assert "local_blocks=0" in source


def test_npu_triton_prefill_calls_prefill_kernels_with_appended_forced_blocks():
    module = _load_npu_triton_prefill_module()
    calls = {}

    def fake_index_kernel(**kwargs):
        calls["index"] = kwargs
        topk_idx = torch.tensor([[[0], [1]]], dtype=torch.int32)
        return None, topk_idx

    def fake_main_kernel(**kwargs):
        calls["main"] = kwargs
        return kwargs["q"].new_full(kwargs["q"].shape, 7)

    module.flash_prefill_npu_with_topk_index = fake_index_kernel
    module.flash_prefill_npu_with_gqa_share_sparse = fake_main_kernel

    q = torch.zeros((2, 1, 1), dtype=torch.bfloat16)
    idx_q = torch.zeros((2, 1, 1), dtype=torch.bfloat16)
    cache = torch.zeros((4, 1, 1), dtype=torch.bfloat16)
    req_to_token = torch.arange(4, dtype=torch.int32).view(1, 4)
    slot_ids = torch.tensor([0], dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    prefix_lens = torch.tensor([2], dtype=torch.int32)

    idx_o, o = module.minimax_sparse_prefill_npu_triton(
        q=q,
        k_cache=cache,
        v_cache=cache,
        sink=None,
        idx_q=idx_q,
        idx_k_cache=cache,
        idx_v_cache=None,
        idx_sink=None,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=2,
        max_seqlen_k=4,
        block_size_q=1,
        block_size_k=2,
        topk=1,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
    )

    assert idx_o is None
    torch.testing.assert_close(o, torch.full_like(q, 7))
    assert calls["index"]["init_blocks"] == 0
    assert calls["index"]["local_blocks"] == 0
    assert "block_table" not in calls["main"]
    expected_topk = torch.tensor([[[0, 1], [1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(calls["main"]["topk_idx"], expected_topk)


def test_npu_triton_prefill_debug_logs_every_call():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
    ).read_text()

    assert "_dbg_prefill_main_diff_count" not in source
    assert "_dbg_prefill_index_diff_count" not in source
    assert "_dbg_prefill_index_skip_count" not in source
    assert "index diff skipped" in source
    assert "disable_index_value=%s" in source
    assert "max_diff_at=%s" in source
    assert "eff_seq_len" in source


def test_npu_triton_prefill_does_not_keep_decode_topk_debug_switch():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill.py"
    ).read_text()

    assert "MINIMAX_NPU_TRITON_PREFILL_TORCH_TOPK" not in source
    assert "use_triton_topk=not _use_torch_topk" not in source


def test_npu_triton_prefill_score_kernel_caps_n_tile_for_ub():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill.py"
    ).read_text()

    assert "_PREFILL_NPU_SCORE_BLOCK_SIZE_N = 64" in source
    assert '"BLOCK_SIZE_N": lambda args: _PREFILL_NPU_SCORE_BLOCK_SIZE_N' in source


def test_npu_triton_prefill_merge_matches_decode_local_only_semantics():
    module = _load_npu_triton_prefill_module()
    topk_idx = torch.tensor(
        [
            [
                [2, 0],
                [0, 1],
                [1, -1],
            ]
        ],
        dtype=torch.int32,
    )
    query_seq_lens = torch.tensor([1, 3, 5], dtype=torch.int32)

    merged = module._merge_prefill_sparse_blocks(
        topk_idx,
        query_seq_lens,
        block_size=2,
        init_blocks=0,
        local_blocks=1,
    )

    expected = torch.tensor(
        [
            [
                [-1, 0, -1],
                [0, 1, -1],
                [1, -1, 2],
            ]
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(merged, expected)


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
