import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.qsa import qsa_indexer as qsa_indexer_module
from sglang.srt.layers.attention.qsa import dsa_indexer as dsa_indexer_module
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    torch_expand_qsa_block_indices,
    triton_expand_qsa_block_indices,
    qsa_fast_topk,
    qsa_sparse_attention,
)
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
)
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1


def test_qwen4_exp_indexer_config_is_read_from_text_config():
    config = Qwen4ExpConfig(
        text_config={
            "hc_count": 4,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": TOKEN_TOPK,
            "indexer_compress_ratio": COMPRESS_RATIO,
        },
    )
    assert config.text_config.indexer_budget == TOKEN_TOPK
    assert config.text_config.indexer_compress_ratio == COMPRESS_RATIO


def _compressed_config_namespace(**overrides):
    fields = dict(
        model_type="qwen4_exp",
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=TOKEN_TOPK,
        indexer_compress_ratio=COMPRESS_RATIO,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _tokenwise_config_namespace(**overrides):
    fields = dict(
        model_type="qwen3_5",
        index_topk=2048,
        index_n_heads=64,
        index_kv_heads=1,
        index_head_dim=128,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_qsa_profile_parses_compressed_qwen4_exp_schema():
    from sglang.srt.layers.attention.qsa.config import (
        QSA_ROPE_MROPE,
        QSA_VARIANT_COMPRESSED,
        is_qwen_qsa,
        parse_qsa_profile,
    )

    wrapped = Qwen4ExpConfig(
        text_config={
            "hc_count": 4,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "indexer_n_heads": 8,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": TOKEN_TOPK,
            "indexer_compress_ratio": COMPRESS_RATIO,
        },
    )
    for config in (wrapped, _compressed_config_namespace()):
        profile = parse_qsa_profile(config)
        assert profile.variant == QSA_VARIANT_COMPRESSED
        assert profile.n_heads == 8
        assert profile.kv_heads == 1
        assert profile.head_dim == 128
        assert profile.budget == TOKEN_TOPK
        assert profile.compress_ratio == COMPRESS_RATIO
        assert profile.block_topk == BLOCK_TOPK
        assert profile.rope_mode == QSA_ROPE_MROPE
        assert profile.draft_extend_cuda_graph is False
        assert is_qwen_qsa(config)
    # The legacy backend module keeps re-exporting the shared detector.
    assert qsa_backend_module.is_qwen_qsa is is_qwen_qsa
    assert not is_qwen_qsa(SimpleNamespace())
    assert parse_qsa_profile(None) is None


def test_qsa_profile_rejects_malformed_compressed_schema():
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile

    bad_configs = {
        "missing": _compressed_config_namespace(indexer_budget=None),
        "ratio_one": _compressed_config_namespace(indexer_compress_ratio=1),
        "indivisible": _compressed_config_namespace(indexer_budget=TOKEN_TOPK - 2),
        "bad_block_topk": _compressed_config_namespace(
            indexer_budget=1024, indexer_compress_ratio=4
        ),
        "kv_heads": _compressed_config_namespace(indexer_kv_heads=2),
    }
    for name, config in bad_configs.items():
        try:
            parse_qsa_profile(config)
        except ValueError:
            continue
        raise AssertionError(f"{name} compressed config must be rejected")


def test_qsa_profile_parses_tokenwise_qwen_schema():
    from sglang.srt.layers.attention.qsa.config import (
        QSA_ROPE_PLAIN,
        QSA_VARIANT_TOKENWISE,
        is_qwen_qsa,
        parse_qsa_profile,
    )

    profile = parse_qsa_profile(_tokenwise_config_namespace())
    assert profile.variant == QSA_VARIANT_TOKENWISE
    assert profile.n_heads == 64
    assert profile.kv_heads == 1
    assert profile.head_dim == 128
    assert profile.budget == 2048
    assert profile.compress_ratio == 1
    assert profile.block_topk == 2048
    assert profile.rope_mode == QSA_ROPE_PLAIN
    assert profile.draft_extend_cuda_graph is False
    assert is_qwen_qsa(_tokenwise_config_namespace())

    for name, config in {
        "budget": _tokenwise_config_namespace(index_topk=1024),
        "kv_heads": _tokenwise_config_namespace(index_kv_heads=2),
        "missing": _tokenwise_config_namespace(index_head_dim=None),
    }.items():
        try:
            parse_qsa_profile(config)
        except ValueError:
            continue
        raise AssertionError(f"{name} tokenwise config must be rejected")


def test_qsa_profile_does_not_swallow_deepseek_nsa_schema():
    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa, parse_qsa_profile

    # DeepSeek NSA configs also expose index_topk; they must keep their own
    # backend/draft paths instead of being parsed as Qwen tokenwise QSA.
    deepseek_like = SimpleNamespace(
        model_type="deepseek_v32",
        index_topk=2048,
        index_n_heads=64,
        index_kv_heads=1,
        index_head_dim=128,
    )
    assert parse_qsa_profile(deepseek_like) is None
    assert not is_qwen_qsa(deepseek_like)

    ambiguous = _compressed_config_namespace(**_tokenwise_config_namespace().__dict__)
    try:
        parse_qsa_profile(ambiguous)
    except ValueError as exc:
        assert "Ambiguous" in str(exc)
    else:
        raise AssertionError("QSA must reject configs with both schemas set")


def test_qsa_glue_builds_indexer_per_variant(monkeypatch):
    from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

    recorded = {}

    class _FakeIndexer:
        def __init__(self, config, layer_id, quant_config=None, prefix="", rotary_emb=None):
            recorded.update(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=prefix,
                rotary_emb=rotary_emb,
            )

    monkeypatch.setattr(qsa_indexer_module, "QSAIndexer", _FakeIndexer)
    rotary = object()
    config = _compressed_config_namespace()
    indexer = build_qsa_indexer(
        config, layer_id=7, quant_config="qc", prefix="p", rotary_emb=rotary
    )
    assert isinstance(indexer, _FakeIndexer)
    assert recorded == dict(
        config=config, layer_id=7, quant_config="qc", prefix="p", rotary_emb=rotary
    )

    # Tokenwise configs build the Lightning Indexer through the same glue.
    class _FakeDSAIndexer:
        def __init__(self, config, layer_id, quant_config=None, prefix="", **kw):
            recorded.update(
                dsa_config=config,
                dsa_layer_id=layer_id,
                dsa_quant_config=quant_config,
                dsa_prefix=prefix,
            )

    monkeypatch.setattr(
        dsa_indexer_module, "QwenDSAIndexer", _FakeDSAIndexer
    )
    dsa_indexer = build_qsa_indexer(
        _tokenwise_config_namespace(), layer_id=2, prefix="q"
    )
    assert isinstance(dsa_indexer, _FakeDSAIndexer)
    assert recorded["dsa_layer_id"] == 2
    assert recorded["dsa_prefix"] == "q"

    try:
        build_qsa_indexer(SimpleNamespace(), layer_id=0, rotary_emb=rotary)
    except ValueError as exc:
        assert "QSA indexer schema" in str(exc)
    else:
        raise AssertionError("non-QSA configs must be rejected")


def test_qsa_glue_fetches_indexer_metadata_without_model_unwrap():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.qsa.glue import get_qsa_indexer_metadata

    full = SimpleNamespace(
        get_indexer_metadata=lambda layer_id, forward_batch: f"meta-{layer_id}",
        token_to_kv_pool=None,
        req_to_token_pool=None,
        needs_cpu_seq_lens=True,
    )
    hybrid = HybridLinearAttnBackend(
        full_attn_backend=full,
        linear_attn_backend=SimpleNamespace(needs_cpu_seq_lens=True),
        full_attn_layers=[3],
    )
    # Full-attention layers fetch through the hybrid wrapper directly.
    assert get_qsa_indexer_metadata(hybrid, 3, object()) == "meta-3"
    # A backend that provides no indexer metadata anywhere must surface an
    # explicit error instead of silently running without sparse selection.
    empty = SimpleNamespace(get_indexer_metadata=lambda layer_id, batch: None)
    try:
        get_qsa_indexer_metadata(empty, 3, object())
    except RuntimeError as exc:
        assert "indexer metadata" in str(exc)
    else:
        raise AssertionError("QSA must fail when no indexer metadata exists")


def test_qsa_cuda_graph_padding_reaches_hybrid_children():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    class Recorder:
        def __init__(self):
            self.num_padding = None
            self.token_to_kv_pool = None
            self.req_to_token_pool = None
            self.needs_cpu_seq_lens = True

        def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
            self.num_padding = getattr(forward_batch, "num_padding", None)

    qsa = Recorder()
    linear = Recorder()
    hybrid_linear = HybridLinearAttnBackend(qsa, linear, full_attn_layers=[0])
    replay_fb = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.zeros(2, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        seq_lens_cpu=torch.ones(2, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        num_padding=1,
    )
    hybrid_linear.init_forward_metadata_out_graph(replay_fb)
    assert qsa.num_padding == 1
    assert linear.num_padding == 1


def test_qsa_draft_extend_backend_decision_follows_profile():
    from sglang.srt.speculative.draft_utils import DraftBackendFactory

    def factory(config):
        return DraftBackendFactory(
            draft_model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=config),
                draft_attention_backend=None,
                attn_backend="draft-runner-backend",
            ),
            topk=1,
            speculative_num_steps=3,
        )

    # Compressed profiles run draft-extend through the draft model runner's
    # own (QSA-wrapped hybrid) backend; the draft-extend CUDA graph replays
    # it with variable accepted rows padded to the captured width.
    assert (
        factory(_compressed_config_namespace()).create_draft_extend_backend()
        == "draft-runner-backend"
    )
    # Tokenwise profiles stay eager (no graph-stable indexer metadata); they
    # must never silently fall back to a dense backend either.
    assert factory(_tokenwise_config_namespace()).create_draft_extend_backend() is None


def test_qsa_backend_stores_parsed_profile():
    from sglang.srt.layers.attention.qsa.config import QSA_VARIANT_COMPRESSED

    runner, _, _ = _make_qsa_runner_and_pool()
    runner.model_config = SimpleNamespace(
        context_len=64,
        hf_text_config=None,
        hf_config=_compressed_config_namespace(),
    )
    backend = QwenSparseAttnBackend(runner)
    assert backend.qsa_profile is not None
    assert backend.qsa_profile.variant == QSA_VARIANT_COMPRESSED
    assert backend.qsa_profile.compress_ratio == COMPRESS_RATIO
    # The legacy field reads keep their current values.
    assert backend.compress_ratio == COMPRESS_RATIO

    tokenwise_fields = dict(runner.__dict__)
    tokenwise_fields["model_config"] = SimpleNamespace(
        context_len=64, hf_text_config=None, hf_config=_tokenwise_config_namespace()
    )
    tokenwise_runner = SimpleNamespace(**tokenwise_fields)
    tokenwise_backend = QwenSparseAttnBackend(tokenwise_runner)
    # Tokenwise shares the backend; the ratio comes from the profile.
    assert tokenwise_backend.compress_ratio == 1
    try:
        tokenwise_backend._capture_cuda_graph_metadata(
            bs=1,
            num_tokens=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            forward_mode=None,
            spec_info=None,
        )
    except NotImplementedError as exc:
        assert "tokenwise" in str(exc)
    else:
        raise AssertionError("tokenwise CUDA graph capture must fail loudly")


def _make_dsa_pool():
    from sglang.srt.mem_cache.qsa_kv_pool import QwenDSATokenToKVPool

    return QwenDSATokenToKVPool(
        size=128,
        dtype=torch.bfloat16,
        page_size=64,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=[0],
        device="cpu",
        mamba_pool=SimpleNamespace(),
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
        qsa_token_budget=2048,
        enable_memory_saver=False,
        start_layer=0,
    )


def test_qwen_dsa_pool_roundtrip_and_layer_mapping():
    from sglang.srt.mem_cache.qsa_kv_pool import QwenDSATokenToKVPool

    pool = _make_dsa_pool()
    assert pool.qsa_compress_ratio == 1
    assert pool.qsa_block_topk == 2048
    assert pool.qsa_token_topk == 2048

    rows = torch.arange(32, dtype=torch.bfloat16).reshape(4, 1, 8)
    loc = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    pool.set_dsa_index_k_buffer(0, loc, rows)
    buffer = pool.get_dsa_index_k_buffer(0)
    torch.testing.assert_close(buffer[5].float(), rows[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(buffer[10].float(), rows[3].float(), rtol=0, atol=0)
    written_rows = buffer.abs().sum(dim=(1, 2)) > 0
    assert int(written_rows.sum()) == 4
    assert written_rows[loc.long()].all()
    assert not written_rows[:5].any()

    try:
        pool.get_dsa_index_k_buffer(7)
    except ValueError as exc:
        assert "full attention layers" in str(exc)
    else:
        raise AssertionError("QwenDSA pool must reject non-full-attention layers")

    try:
        QwenDSATokenToKVPool(
            size=128,
            dtype=torch.bfloat16,
            page_size=32,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=[0],
            device="cpu",
            mamba_pool=SimpleNamespace(),
            qsa_index_kv_heads=1,
            qsa_index_head_dim=8,
            qsa_token_budget=2048,
        )
    except ValueError as exc:
        assert "page_size 64" in str(exc)
    else:
        raise AssertionError("QwenDSA pool must require page_size 64")


def test_mambaish_pool_profile_variants_parse():
    from sglang.srt.layers.attention.qsa.config import (
        QSA_VARIANT_COMPRESSED,
        QSA_VARIANT_TOKENWISE,
        parse_qsa_profile,
    )

    assert parse_qsa_profile(None) is None
    assert (
        parse_qsa_profile(_compressed_config_namespace()).variant
        == QSA_VARIANT_COMPRESSED
    )
    assert (
        parse_qsa_profile(_tokenwise_config_namespace()).variant
        == QSA_VARIANT_TOKENWISE
    )


def test_qwen_dsa_indexer_rejects_non_tokenwise_configs():
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    for config in (
        _compressed_config_namespace(),
        SimpleNamespace(hidden_size=32),
        _tokenwise_config_namespace(index_topk=1024),
    ):
        try:
            QwenDSAIndexer(config, layer_id=0)
        except ValueError:
            continue
        raise AssertionError("QwenDSAIndexer must reject invalid configs")


def test_qwen_dsa_weighted_mqa_logits_matches_explicit_formula():
    torch.manual_seed(3)
    q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    w = torch.randn(3, 4, dtype=torch.bfloat16)
    k = torch.randn(5, 1, 8, dtype=torch.bfloat16)
    actual = dsa_indexer_module.torch_dsa_weighted_mqa_logits(q, w, k, 2.0)
    scores = torch.einsum("mhd,nhd->mnh", q.float(), k.float())
    expected = torch.relu(scores)
    expected = (expected * w.float().unsqueeze(1)).sum(-1) / 2.0
    torch.testing.assert_close(actual, expected)


def _make_dsa_indexer_stub(topk=4):
    return SimpleNamespace(
        layer_id=0,
        token_topk=topk,
        score_scale=1.0,
        index_n_heads=2,
    )


def test_qwen_dsa_select_paged_picks_per_row_windows():
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    pool = SimpleNamespace()
    # score(row, col) = relu(k[col, 0, 0]) * 2 (see q/w below).
    index_k = torch.zeros(8, 1, 4, dtype=torch.bfloat16)
    # row 0 sees slots 1,2,3 (len 3): scores 10, 6, 2 → picks cols 0,1,2.
    index_k[1, 0, 0] = 5
    index_k[2, 0, 0] = 3
    index_k[3, 0, 0] = 1
    # row 1 sees slots 2,4,6,7 (len 4): col 3 (slot 7) scores highest.
    index_k[7, 0, 0] = 9
    pool.get_dsa_index_k_buffer = lambda layer_id: index_k
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([3, 4], dtype=torch.int32),
        token_slot_table=torch.tensor(
            [[1, 2, 3, 4, 5, 6], [2, 4, 6, 7, 0, 1]], dtype=torch.int32
        ),
        token_to_batch_idx=torch.arange(2, dtype=torch.int32),
        token_to_kv_pool=pool,
    )
    indexer = _make_dsa_indexer_stub(topk=4)
    q = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    q[..., 1:] = 0
    w = torch.ones(2, 2, dtype=torch.bfloat16)
    out = QwenDSAIndexer._select_paged(indexer, q, w, metadata)
    assert out.shape == (2, 4)
    assert sorted(out[0, :3].tolist()) == [0, 1, 2]
    assert out[0, 3].item() == -1
    assert sorted(out[1].tolist()) == [0, 1, 2, 3]


def test_qwen_dsa_select_prefill_respects_causal_windows():
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    index_k = torch.zeros(16, 1, 4, dtype=torch.bfloat16)
    # seq 0 packed slot 4 is its logical col 1; seq 1 packed slots are
    # [8, 9, 10, 11, 12] → slot 9 is its logical col 1, slot 12 its col 4.
    index_k[4, 0, 0] = 8
    index_k[9, 0, 0] = 8
    index_k[12, 0, 0] = 20
    pool = SimpleNamespace(get_dsa_index_k_buffer=lambda layer_id: index_k)
    token_slot_table = torch.stack(
        [torch.arange(3, 11, dtype=torch.int32), torch.arange(8, 16, dtype=torch.int32)]
    )
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([4, 5], dtype=torch.int32),
        token_slot_table=token_slot_table,
        token_to_batch_idx=torch.tensor(
            [0, 0, 0, 1, 1, 1, 1], dtype=torch.int32
        ),
        token_to_kv_pool=pool,
    )
    indexer = _make_dsa_indexer_stub(topk=4)
    q = torch.ones(7, 2, 4, dtype=torch.bfloat16)
    q[..., 1:] = 0
    w = torch.ones(7, 2, dtype=torch.bfloat16)
    # seq 0 extends positions 1..3, seq 1 extends positions 1..4.
    logical_positions = torch.tensor(
        [1, 2, 3, 1, 2, 3, 4], dtype=torch.int64
    )
    out = QwenDSAIndexer._select_prefill(
        indexer, q, w, logical_positions, metadata
    )
    assert out.shape == (7, 4)
    # A row may never select a column beyond its causal end — in particular
    # seq 1 rows before position 4 must not see the hot col 4.
    for row, position in enumerate(logical_positions.tolist()):
        valid = out[row][out[row] >= 0]
        assert int(valid.max() if valid.numel() else 0) <= position
    # The hot columns show up once their windows open.
    assert 1 in out[0].tolist()  # seq 0
    assert 1 in out[3].tolist()  # seq 1
    assert 4 in out[6].tolist()  # seq 1 at position 4


def test_qwen_dsa_indexer_forward_trims_dp_padding_rows():
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    calls = {}
    pool = SimpleNamespace(
        set_dsa_index_k_buffer=lambda layer_id, loc, k: calls.update(
            locs=loc.clone(), k_rows=k.shape[0]
        )
    )
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(1, dtype=torch.int32),
        get_seqlens_expanded=lambda: torch.tensor([7], dtype=torch.int32),
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        token_slot_table=torch.arange(8, dtype=torch.int32).reshape(1, 8),
        token_to_batch_idx=torch.zeros(1, dtype=torch.int32),
        token_to_kv_pool=pool,
        out_cache_loc=torch.tensor([10, 11], dtype=torch.int32),
    )
    indexer = SimpleNamespace(
        layer_id=0,
        token_topk=4,
        score_scale=1.0,
        index_n_heads=2,
        project_qkw=lambda hidden, positions: (
            torch.zeros(positions.shape[0], 2, 4),
            torch.zeros(positions.shape[0], 2),
            torch.zeros(positions.shape[0], 1, 4),
        ),
        _select_paged=lambda q, w, metadata: calls.update(q_rows=q.shape[0])
        or torch.empty(q.shape[0], 4, dtype=torch.int32),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        positions=torch.tensor([6, 0]),
        out_cache_loc=torch.tensor([10, 11], dtype=torch.int32),
    )
    hidden = torch.zeros(2, 16)
    QwenDSAIndexer.forward_cuda(
        indexer, hidden, forward_batch.positions, forward_batch, metadata
    )
    # Only the semantic row is projected, stored and selected; the padded row
    # never reaches the pool or the selector.
    assert calls["locs"].tolist() == [10]
    assert calls["k_rows"] == 1
    assert calls["q_rows"] == 1


def test_qwen_dsa_indexer_forward_empty_batch_returns_zero_rows():
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(0, dtype=torch.int32),
        get_seqlens_expanded=lambda: torch.zeros(0, dtype=torch.int32),
    )
    indexer = _make_dsa_indexer_stub()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.IDLE,
        positions=torch.zeros(0, dtype=torch.int64),
        out_cache_loc=torch.zeros(0, dtype=torch.int32),
    )
    out = QwenDSAIndexer.forward_cuda(
        indexer,
        torch.zeros(0, 16),
        forward_batch.positions,
        forward_batch,
        metadata,
    )
    assert out.shape == (0, 4)


def test_qsa_backend_is_registered():
    assert ATTENTION_BACKENDS["qsa"] is not None


def test_qsa_idle_skips_metadata_construction():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.forward_metadata = object()

    backend.init_forward_metadata(
        SimpleNamespace(forward_mode=ForwardMode.IDLE)
    )

    assert backend.forward_metadata is None


def _make_mtp_draft_batch(steps: int, seq_lens=(8, 16), loc_base: int = 40):
    bs = len(seq_lens)
    pool = _FakeQSAPool(capacity=bs * steps + 256)
    req_to_token = torch.stack(
        [torch.arange(i * 64, (i + 1) * 64, dtype=torch.int32) for i in range(bs)]
    )
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=64,
            hf_config=SimpleNamespace(indexer_compress_ratio=COMPRESS_RATIO),
        ),
    )
    backend = QwenSparseMultiStepDraftBackend(
        runner, topk=1, speculative_num_steps=steps
    )
    full_out_cache_loc = loc_base + torch.arange(bs * steps, dtype=torch.int32)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=runner.req_to_token_pool,
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        positions=torch.tensor([length - 1 for length in seq_lens], dtype=torch.int64),
        out_cache_loc=full_out_cache_loc,
        forward_mode=ForwardMode.DECODE,
    )
    return backend, forward_batch, pool


def test_qsa_cuda_graph_pads_dynamic_draft_extend_rows():
    row_lengths, row_req_pool_indices, row_prefix_lengths = (
        QwenSparseAttnBackend._graph_speculative_layout(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([11, 21], dtype=torch.int32),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=SimpleNamespace(extend_seq_lens_cpu=[1, 2]),
        )
    )
    assert row_lengths.tolist() == [11, 20, 21, 1, 1, 1, 1, 1]
    assert row_req_pool_indices.tolist() == [3, 7, 7, 3, 3, 3, 3, 3]
    assert row_prefix_lengths.tolist() == [10, 19, 19, 0, 0, 0, 0, 0]


def test_qsa_cuda_graph_target_verify_ignores_capture_bucket_requests():
    row_lengths, row_req_pool_indices, row_prefix_lengths = (
        QwenSparseAttnBackend._graph_speculative_layout(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([3, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([9, 1], dtype=torch.int32),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=4),
            num_padding=1,
        )
    )
    assert row_lengths.tolist() == [10, 11, 12, 13, 1, 1, 1, 1]
    assert row_req_pool_indices.tolist() == [3] * 8
    assert row_prefix_lengths.tolist() == [9, 9, 9, 9, 0, 0, 0, 0]


def test_qsa_target_verify_rejects_branching_speculation():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.compress_ratio = 4
    try:
        backend._require_chain_speculation(
            ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=2)
        )
    except NotImplementedError as exc:
        assert "topk=1" in str(exc)
    else:
        raise AssertionError("QSA target verification must reject tree branches")
    # The pending-group ring keys state by position % ratio: a verify window
    # wider than the ratio would collide within one forward.
    try:
        backend._require_chain_speculation(
            ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=1, draft_token_num=5)
        )
    except NotImplementedError as exc:
        assert "compress ratio" in str(exc)
    else:
        raise AssertionError("QSA must reject draft windows wider than the ratio")
    backend._require_chain_speculation(
        ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=1, draft_token_num=4)
    )


def test_qsa_mtp_cuda_graph_padding_stays_below_compression_boundary():
    backend, forward_batch, _ = _make_mtp_draft_batch(
        steps=4, seq_lens=(3, 1)
    )

    class Recorder:
        def __init__(self):
            self.capture_lengths = None
            self.replay_lengths = None

        def _capture_cuda_graph_metadata(self, **kwargs):
            self.capture_lengths = kwargs["seq_lens"].clone()

        def _replay_cuda_graph_metadata(self, *, bs, seq_lens, **kwargs):
            self.replay_lengths = seq_lens[:bs].clone()

    recorders = [Recorder() for _ in backend.attn_backends]
    backend.attn_backends = recorders
    forward_batch.spec_info = SimpleNamespace()

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    forward_batch.num_padding = 1
    backend.init_forward_metadata_out_graph(forward_batch)

    assert [r.capture_lengths.tolist() for r in recorders] == [[1, 1]] * 3
    assert [r.replay_lengths.tolist() for r in recorders] == [
        [4, 1],
        [5, 1],
        [6, 1],
    ]


def test_qsa_indexer_ignores_dp_attention_token_padding():
    calls = {}

    def run(num_rows):
        mapping = torch.zeros(15, dtype=torch.int32)
        metadata = SimpleNamespace(
            token_to_kv_pool=None,
            compress_member_rows=None,
            decode_logical_positions=None,
            pending_ring_slots=None,
            get_token_to_batch_idx=lambda: mapping,
            get_prefill_mqa_inputs=lambda layer_id, logical_positions: (
                torch.empty(0, 1, 128),
                torch.zeros(15, dtype=torch.int32),
                torch.zeros(15, dtype=torch.int32),
                torch.tensor([15], dtype=torch.int32),
            ),
        )
        indexer = SimpleNamespace(
            layer_id=0,
            project_qk=lambda hidden, rope_positions, **kwargs: (
                hidden.reshape(-1, 1, 2),
                hidden.reshape(-1, 1, 2),
                False,
            ),
            _pending_ring_slots=lambda metadata, logical_positions, is_extend: (
                torch.zeros(logical_positions.numel(), dtype=torch.long)
            ),
            update_key_state_and_compress=lambda token_k, logical, rope, meta, state_slots=None, state_stored=False: (
                calls.update(
                    token_rows=token_k.shape[0],
                    logical_rows=logical.numel(),
                    rope_rows=rope.numel(),
                )
            ),
            select_prefill_tokens=lambda q, keys, starts, ends, logical, lengths: (
                q.squeeze(1)
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            positions=torch.cat(
                [torch.arange(15), torch.zeros(num_rows - 15, dtype=torch.long)]
            ),
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32),
        )
        hidden = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)
        return QSAIndexer.forward_cuda(
            indexer, hidden, forward_batch.positions, forward_batch, metadata
        )

    unpadded = run(15)
    padded = run(16)
    torch.testing.assert_close(padded, unpadded)
    assert padded.shape[0] == 15
    assert calls == {"token_rows": 15, "logical_rows": 15, "rope_rows": 15}


def test_qsa_extend_matches_with_and_without_dp_attention_padding(monkeypatch):
    kernel_q_rows = []

    def fake_sparse_attention(q, k, v, slots, scale):
        kernel_q_rows.append(q.shape[0])
        return q + 1

    monkeypatch.setattr(
        qsa_backend_module, "qsa_sparse_attention", fake_sparse_attention
    )
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.zeros(15, dtype=torch.int32),
        sequence_lengths=torch.tensor([15], dtype=torch.int32),
        token_slot_table=torch.arange(16, dtype=torch.int32).reshape(1, 16),
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.forward_metadata = metadata

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            pass

        def get_key_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

        def get_value_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

    pool = Pool()
    backend.token_to_kv_pool = pool
    layer = SimpleNamespace(tp_q_head_num=1, head_dim=2, layer_id=0, scaling=1.0)
    topk = torch.zeros(15, 3, dtype=torch.int32)

    def run(num_rows):
        values = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=pool,
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32),
        )
        return backend.forward_extend(
            values, values.clone(), values.clone(), layer, batch, topk_indices=topk
        )

    unpadded = run(15)
    padded = run(16)
    torch.testing.assert_close(padded[:15], unpadded)
    torch.testing.assert_close(padded[15], torch.zeros(2))
    assert padded.shape == (16, 2)
    assert kernel_q_rows == [15, 15]


def test_qsa_cuda_extend_ignores_dp_attention_padding(monkeypatch):
    if not torch.cuda.is_available():
        return

    kernel_shapes = []

    def fake_sparse_gqa(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
        kernel_shapes.append((q.shape[0], k.shape[0], v.shape[0], indices.shape[0]))
        return q + 1

    monkeypatch.setattr(
        qsa_backend_module, "sparse_gqa_fwd_interface_triton", fake_sparse_gqa
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            pass

    pool = Pool()
    backend.token_to_kv_pool = pool
    layer = SimpleNamespace(tp_q_head_num=1, head_dim=2, layer_id=0, scaling=1.0)
    topk = torch.zeros(15, 3, dtype=torch.int32, device="cuda")

    def run(num_rows):
        values = torch.arange(num_rows * 2, dtype=torch.float32, device="cuda").reshape(
            num_rows, 2
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=pool,
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32, device="cuda"),
            extend_seq_lens=torch.tensor([15], dtype=torch.int32, device="cuda"),
            extend_seq_lens_cpu=[15],
            seq_lens_cpu=[15],
        )
        return backend.forward_extend(
            values, values.clone(), values.clone(), layer, batch, topk_indices=topk
        )

    unpadded = run(15)
    padded = run(16)
    torch.testing.assert_close(padded[:15], unpadded)
    torch.testing.assert_close(padded[15], torch.zeros(2, device="cuda"))
    assert kernel_shapes == [(15, 15, 15, 15), (15, 15, 15, 15)]


def _make_paged_extend_backend():
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.zeros(3, dtype=torch.int32),
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        token_slot_table=torch.arange(16, dtype=torch.int32).reshape(1, 16),
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.forward_metadata = metadata

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            pass

        def get_key_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

        def get_value_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

    layer = SimpleNamespace(tp_q_head_num=1, head_dim=2, layer_id=0, scaling=1.0)
    pool = Pool()
    backend.token_to_kv_pool = pool
    return backend, pool, layer


def test_qsa_paged_extend_trims_padding_rows_and_restores_output(monkeypatch):
    kernel_rows = []

    def fake_sparse_attention(q, k, v, slots, scale):
        kernel_rows.append((q.shape[0], slots.shape[0]))
        return q + 1

    monkeypatch.setattr(
        qsa_backend_module, "qsa_sparse_attention", fake_sparse_attention
    )
    backend, pool, layer = _make_paged_extend_backend()
    topk = torch.zeros(3, 2, dtype=torch.int32)

    def run(num_rows, mode):
        values = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)
        batch = SimpleNamespace(
            forward_mode=mode,
            token_to_kv_pool=pool,
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32),
        )
        return backend.forward_extend(
            values, values.clone(), values.clone(), layer, batch, topk_indices=topk
        )

    # DP attention pads the physical q rows past the semantic draft rows in
    # every speculative paged mode; padding must round-trip through the kernel.
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        unpadded = run(3, mode)
        padded = run(5, mode)
        torch.testing.assert_close(padded[:3], unpadded)
        torch.testing.assert_close(padded[3:], torch.zeros(2, 2))
        assert padded.shape == (5, 2)
    # The reference path (and any kernel behind it) only ever saw valid rows.
    assert kernel_rows == [(3, 3)] * 4

    # More semantic rows than physical q rows is a bug and must not silently
    # truncate the top-k table.
    short_q = torch.zeros(2, 2, dtype=torch.float32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        token_to_kv_pool=pool,
        out_cache_loc=torch.arange(2, dtype=torch.int32),
    )
    try:
        backend.forward_extend(
            short_q, short_q.clone(), short_q.clone(), layer, batch, topk_indices=topk
        )
    except ValueError as exc:
        assert "top-k rows exceed query rows" in str(exc)
    else:
        raise AssertionError("QSA must reject top-k rows beyond the query rows")


def test_qsa_paged_extend_never_feeds_mismatched_rows_to_kernels(monkeypatch):
    seen = []

    def fake_paged_attention(self, q, layer, forward_batch, topk_indices):
        # Emulates the FA3/FA2 contracts: every kernel requires one cache
        # row per q row (the repro crashed FA3 with seqused_k 40 vs q 44).
        seen.append((q.shape[0], topk_indices.shape[0]))
        return q.reshape(q.shape[0], -1) + 1

    monkeypatch.setattr(
        QwenSparseAttnBackend, "_forward_paged_attention", fake_paged_attention
    )
    backend, pool, layer = _make_paged_extend_backend()
    topk = torch.zeros(3, 2, dtype=torch.int32)
    values = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        token_to_kv_pool=pool,
        out_cache_loc=torch.arange(5, dtype=torch.int32),
    )
    output = backend.forward_extend(
        values, values.clone(), values.clone(), layer, batch, topk_indices=topk
    )
    assert seen == [(3, 3)]
    assert output.shape == (5, 2)
    torch.testing.assert_close(output[3:], torch.zeros(2, 2))


def test_qsa_indexer_rejects_shorter_source_than_request_mapping():
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(16, dtype=torch.int32)
    )
    indexer = SimpleNamespace()
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=torch.arange(15))
    try:
        QSAIndexer.forward_cuda(
            indexer, torch.zeros(15, 2), batch.positions, batch, metadata
        )
    except ValueError as exc:
        assert "logical positions are shorter" in str(exc)
    else:
        raise AssertionError("QSA must reject a request mapping longer than positions")


def _make_qsa_runner_and_pool(num_reqs=4):
    pool = _FakeQSAPool(capacity=256)
    req_to_token = torch.stack(
        [torch.arange(i * 64, (i + 1) * 64, dtype=torch.int32) for i in range(num_reqs)]
    )
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=64,
            hf_config=SimpleNamespace(indexer_compress_ratio=COMPRESS_RATIO),
        ),
    )
    return runner, pool, runner.req_to_token_pool


def test_qsa_idle_metadata_builds_empty_rows():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    idle_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.empty(0, dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int64),
        positions=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        # The MTP wrapper forwards idle batches as zero-row DECODE steps.
        forward_mode=ForwardMode.DECODE,
    )
    backend = QwenSparseAttnBackend(runner)
    backend.init_forward_metadata(idle_batch)
    metadata = backend.forward_metadata
    assert metadata.sequence_lengths.numel() == 0
    assert metadata.token_to_batch_idx.numel() == 0
    assert metadata.row_req_pool_indices.numel() == 0
    assert metadata.token_slot_table.shape[0] == 0
    assert metadata.indexer_metadata.out_cache_loc.numel() == 0

    # The MTP multi-step wrapper forwards idle batches as zero-row DECODE
    # steps; those must not fall into the empty extend/max() path either.
    draft = QwenSparseMultiStepDraftBackend(runner, topk=1, speculative_num_steps=2)
    draft.init_forward_metadata(idle_batch)
    step_metadata = draft.attn_backends[0].forward_metadata
    assert step_metadata.sequence_lengths.numel() == 0
    assert step_metadata.row_req_pool_indices.numel() == 0
    # Per-step out_cache_loc slicing must stay empty without allocating rows.
    for attn_backend in draft.attn_backends:
        assert (
            attn_backend.forward_metadata.indexer_metadata.out_cache_loc.numel() == 0
        )


def test_qsa_decode_requires_one_query_row_per_request():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([1, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8, 16], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 16], dtype=torch.int32),
        # 2 request rows but 3 query rows: this layout is ambiguous under DP.
        positions=torch.tensor([7, 15, 0], dtype=torch.int64),
        out_cache_loc=torch.arange(3, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    try:
        backend.init_forward_metadata(forward_batch)
    except ValueError as exc:
        assert "exactly one query row per request" in str(exc)
    else:
        raise AssertionError("QSA decode must reject multi-row requests")


def test_qsa_tokenwise_decode_metadata_skips_compressed_precompute():
    """Tokenwise pools carry no compressed-page geometry; the once-per-forward
    decode-MQA precompute is a compressed-variant fast path and must not read
    qsa_compressed_page_size off a tokenwise pool -- an ungated read crashed
    every tokenwise decode forward with AttributeError."""

    class _TokenwisePool:
        qsa_compress_ratio = 1
        qsa_token_topk = 2048
        qsa_block_topk = 2048

    pool = _TokenwisePool()
    req_to_token = torch.stack(
        [torch.arange(i * 64, (i + 1) * 64, dtype=torch.int32) for i in range(4)]
    )
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=64, hf_config=_tokenwise_config_namespace()
        ),
    )
    backend = QwenSparseAttnBackend(runner)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=runner.req_to_token_pool,
        req_pool_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        positions=torch.tensor([7], dtype=torch.int64),
        out_cache_loc=torch.arange(1, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    backend.init_forward_metadata(forward_batch)
    metadata = backend.forward_metadata.indexer_metadata
    assert metadata.decode_page_table is None
    assert metadata.decode_lengths is None


def test_qsa_extend_rope_matrix_uses_mrope_coordinates():
    """VL models swap the layer positions to forward_batch.mrope_positions
    ([3, N] t/h/w); the build-time extend_rope_matrix must be built from that
    source -- building it from the flat 1-D positions RoPEs prefill-compressed
    keys with wrong coordinates while decode-compressed keys of the same
    request use true mrope coords, silently corrupting indexer selection on
    image-bearing sequences."""
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    num_tokens = 8
    flat = torch.arange(num_tokens, dtype=torch.int64)
    mrope = torch.stack([flat, flat + 100, flat + 200])
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
        positions=flat,
        mrope_positions=mrope,
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int32),
        _original_forward_mode=None,
    )
    backend.init_forward_metadata(forward_batch)
    got = backend.forward_metadata.indexer_metadata.extend_rope_matrix
    assert got is not None
    assert torch.equal(got, mrope.transpose(0, 1))


def test_qsa_speculative_pseudo_extend_is_rejected():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    for original_mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        forward_batch = SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=req_pool,
            req_pool_indices=torch.tensor([1], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
            # DP MAX_LEN pseudo-extend rewrites the mode and loses the
            # per-request draft fan-out of the original speculative mode.
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.ones(1, dtype=torch.int32),
            positions=torch.tensor([9], dtype=torch.int64),
            out_cache_loc=torch.arange(1, dtype=torch.int32),
            _original_forward_mode=original_mode,
        )
        try:
            backend.init_forward_metadata(forward_batch)
        except ValueError as exc:
            assert "pseudo-extend" in str(exc)
        else:
            raise AssertionError(
                f"QSA must reject pseudo-extend of {original_mode}"
            )


def test_qsa_indexer_requires_compress_ratio_above_one():
    try:
        QSAIndexer._validate_config(
            SimpleNamespace(
                indexer_n_heads=8,
                indexer_kv_heads=1,
                indexer_head_dim=64,
                indexer_budget=512,
                indexer_compress_ratio=1,
            )
        )
    except ValueError as exc:
        assert "compress_ratio" in str(exc)
    else:
        raise AssertionError("QSA must reject a unit compression ratio")


def test_qsa_indexer_decode_ignores_dp_attention_token_padding():
    calls = {}
    mapping = torch.zeros(1, dtype=torch.int32)
    metadata = SimpleNamespace(
        token_to_kv_pool=None,
        compress_member_rows=None,
        decode_logical_positions=None,
        pending_ring_slots=None,
        get_token_to_batch_idx=lambda: mapping,
        get_seqlens_expanded=lambda: torch.tensor([7], dtype=torch.int32),
        get_seqlens_int32=lambda: torch.tensor([8], dtype=torch.int32),
        get_decode_mqa_inputs=lambda layer_id: (
            torch.zeros(1, 2, 1, 2),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            2,
        ),
    )
    indexer = SimpleNamespace(
        layer_id=0,
        index_n_heads=4,
        _pending_ring_slots=lambda metadata, logical_positions, is_extend: (
            torch.zeros(logical_positions.numel(), dtype=torch.long)
        ),
        project_qk=lambda hidden, rope_positions, **kwargs: (
            hidden.reshape(-1, 1, 2),
            hidden.reshape(-1, 1, 2),
            False,
        ),
        update_key_state_and_compress=lambda token_k, logical, rope, meta, state_slots=None, state_stored=False: (
            calls.update(
                token_rows=token_k.shape[0],
                logical_rows=logical.numel(),
                rope_rows=rope.numel(),
            )
        ),
        select_decode_tokens=lambda q, cache, table, lengths, max_len, positions, seqlens: (
            calls.update(q_rows=q.shape[0], positions=positions.tolist())
            or q.squeeze(1)
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        # One real decode position plus one DP padding row.
        positions=torch.tensor([6, 0]),
        out_cache_loc=torch.arange(2, dtype=torch.int32),
    )
    hidden = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    QSAIndexer.forward_cuda(
        indexer, hidden, forward_batch.positions, forward_batch, metadata
    )
    assert calls == {
        "token_rows": 1,
        "logical_rows": 1,
        "rope_rows": 1,
        "q_rows": 1,
        "positions": [6],
    }


class _FakeQSAPool:
    qsa_index_kv_heads = 1
    qsa_index_head_dim = 128
    qsa_compressed_page_size = 64
    qsa_compress_ratio = COMPRESS_RATIO
    qsa_token_topk = TOKEN_TOPK
    qsa_block_topk = BLOCK_TOPK

    def __init__(self, capacity=32):
        self.token_k = torch.zeros(capacity, 1, 128, dtype=torch.bfloat16)
        self.compressed_k = torch.zeros(3 * 64, 1, 128, dtype=torch.bfloat16)
        self.mapping = torch.full((capacity,), -1, dtype=torch.int32)
        self.qsa_index_loc_map = self.mapping
        self.qsa_page_loc_map = torch.full((capacity,), -1, dtype=torch.int32)
        self.qsa_free_index_slots = torch.empty(0, dtype=torch.int32)
        self.rope_positions = torch.zeros(capacity, 3, dtype=torch.int64)
        self.qsa_page_table = None
        self.qsa_page_owned = None
        self.next_page = 1
        self.bulk_alloc_calls = 0

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.token_k[loc.long()] = token_k

    def get_qsa_key_state_buffer(self, layer_id):
        return self.token_k

    def clear_qsa_key_state_buffer(self, layer_id, loc):
        self.token_k[loc.long()] = 0

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.rope_positions[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc):
        return self.rope_positions[loc.long()]

    def get_or_alloc_qsa_index_locs(self, full_locs, page_anchor_locs, page_offsets):
        result = torch.empty_like(full_locs, dtype=torch.int32)
        for index, (full_loc, anchor, offset) in enumerate(
            zip(full_locs.long(), page_anchor_locs.long(), page_offsets.int())
        ):
            if self.mapping[anchor] < 0:
                self.mapping[anchor] = self.next_page * 64
                self.next_page += 1
            result[index] = self.mapping[anchor] + offset
            self.mapping[full_loc] = result[index]
        return result

    def alloc_qsa_compressed_pages(self, num_pages):
        self.bulk_alloc_calls += 1
        pages = torch.arange(
            self.next_page, self.next_page + num_pages, dtype=torch.int32
        )
        self.next_page += num_pages
        return pages

    def copy_qsa_compressed_page_prefixes(
        self, source_pages, destination_pages, copy_blocks
    ):
        for source, destination, blocks in zip(
            source_pages.tolist(), destination_pages.tolist(), copy_blocks.tolist()
        ):
            src = source * 64
            dst = destination * 64
            self.compressed_k[dst : dst + blocks] = self.compressed_k[
                src : src + blocks
            ]

    def alloc_qsa_compressed_page(self, source_page=-1, copy_blocks=0):
        page = int(self.alloc_qsa_compressed_pages(1)[0])
        if source_page >= 0 and copy_blocks > 0:
            self.copy_qsa_compressed_page_prefixes(
                torch.tensor([source_page]),
                torch.tensor([page]),
                torch.tensor([copy_blocks]),
            )
        return page

    def set_qsa_index_locs(self, full_locs, page, page_offsets):
        result = page * 64 + page_offsets.int()
        self.mapping[full_locs.long()] = result
        return result

    def get_qsa_index_locs(self, full_locs):
        result = self.mapping[full_locs.long()]
        assert torch.all(result >= 0)
        return result

    def get_qsa_compressed_page_locs(self, page_anchor_locs):
        return self.get_qsa_index_locs(page_anchor_locs) // 64

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed_k[loc.long()] = compressed_k

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed_k


class _DispatchIndexer:
    layer_id = 3
    index_n_heads = 4
    compress_ratio = 4
    _pending_ring_slots = QSAIndexer._pending_ring_slots

    def __init__(self):
        self.selected = None
        self.logical_positions = None

    @staticmethod
    def project_qk(hidden_states, positions, **kwargs):
        rows = hidden_states.shape[0]
        return torch.zeros(rows, 4, 128), torch.zeros(rows, 1, 128), False

    def select_prefill_tokens(self, *args):
        self.selected = "prefill"
        return torch.tensor([1])

    def select_decode_tokens(self, *args):
        self.selected = "decode"
        return torch.tensor([2])

    def update_key_state_and_compress(
        self, token_k, logical_positions, rope_positions, metadata, **kwargs
    ):
        self.logical_positions = logical_positions.clone()


class _DispatchMetadata:
    token_to_kv_pool = None
    out_cache_loc = None
    compress_member_rows = None
    decode_logical_positions = None
    pending_ring_slots = None
    # Consumed by the real _pending_ring_slots helper the dispatch indexer
    # borrows: one token row owned by request slot 1.
    token_to_batch_idx = torch.zeros(2, dtype=torch.int32)
    req_pool_indices = torch.ones(2, dtype=torch.int32)
    sequence_lengths = torch.ones(2, dtype=torch.int32)

    @staticmethod
    def get_decode_mqa_inputs(layer_id):
        return (
            torch.zeros(2, 1, 1, 128),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
        )

    @staticmethod
    def get_prefill_mqa_inputs(layer_id, positions):
        return (
            torch.zeros(1, 1, 128),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    @staticmethod
    def get_token_to_batch_idx():
        return torch.zeros(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_int32():
        return torch.ones(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_expanded():
        return torch.tensor([7], dtype=torch.int32)


class _ForwardMode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


def test_qsa_indexer_validates_config_values_instead_of_fixed_shapes():
    QSAIndexer._validate_config(
        SimpleNamespace(
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_head_dim=64,
            indexer_budget=1024,
            indexer_compress_ratio=2,
        )
    )


def test_qsa_average_pool_matches_training_reference():
    token_keys = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 1, 128)
    expected = token_keys.reshape(2, 4, 1, 128).mean(dim=1)
    actual = average_pool_qsa_keys(token_keys.to(torch.bfloat16).reshape(2, 4, 1, 128))
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=2)


def test_qsa_row_ranges_do_not_cross_sequences():
    sequence_lengths = torch.tensor([10, 7], dtype=torch.int32)
    query_positions = torch.tensor([8, 9, 4, 6], dtype=torch.int32)
    query_sequence_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    starts, ends, compressed_cu = build_qsa_row_ranges(
        sequence_lengths, query_positions, query_sequence_ids, COMPRESS_RATIO
    )
    assert compressed_cu.tolist() == [0, 2, 3]
    assert starts.tolist() == [0, 0, 2, 2]
    assert ends.tolist() == [2, 2, 3, 3]


def test_qsa_weight_free_mqa_logits_matches_explicit_formula():
    torch.manual_seed(1)
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(5, 1, 128, dtype=torch.bfloat16)
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    ends = torch.tensor([2, 4, 5], dtype=torch.int32)
    actual = qsa_mqa_prefill(q, k, starts, ends)
    broadcast_k = k.expand(-1, 4, -1)
    expected = torch.einsum("mhd,nhd->mnh", q.float(), broadcast_k.float())
    expected = torch.relu(expected).sum(-1) / (128**0.5)
    columns = torch.arange(5).unsqueeze(0)
    expected.masked_fill_(
        (columns < starts[:, None]) | (columns >= ends[:, None]), -float("inf")
    )
    torch.testing.assert_close(actual, expected)


def test_qsa_prefill_selection_microchunks_rows(monkeypatch):
    rows, keys, heads, head_dim = 65, 64, 4, 8
    token_topk, compress_ratio = 8, 4
    indexer = SimpleNamespace(
        token_topk=token_topk,
        compress_ratio=compress_ratio,
        block_topk=token_topk // compress_ratio,
    )
    torch.manual_seed(2)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, head_dim, dtype=torch.bfloat16)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), keys, dtype=torch.int32)
    positions = torch.full((rows,), keys * compress_ratio - 1, dtype=torch.long)
    sequence_lengths = torch.full(
        (rows,), keys * compress_ratio, dtype=torch.int32
    )

    monkeypatch.setattr(
        qsa_indexer_module,
        "_QSA_PREFILL_LOGITS_BUDGET_BYTES",
        32 * keys * torch.float32.itemsize,
    )
    assert qsa_indexer_module._qsa_prefill_row_chunk_size(rows, keys, heads) == 32
    actual = QSAIndexer.select_prefill_tokens(
        indexer, q, k, starts, ends, positions, sequence_lengths
    )

    logits = qsa_mqa_prefill(q, k, starts, ends)
    blocks = qsa_fast_topk(logits, starts, ends, topk=indexer.block_topk)
    expected = expand_qsa_block_indices(
        blocks,
        positions,
        sequence_lengths,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
    )
    torch.testing.assert_close(actual, expected)


def test_qsa_forward_cuda_dispatches_prefill_and_decode_mqa():
    indexer = _DispatchIndexer()
    metadata = _DispatchMetadata()
    inputs = torch.zeros(1, 16)
    positions = torch.zeros(1, dtype=torch.int32)

    prefill_result = QSAIndexer.forward_cuda(
        indexer,
        inputs,
        positions,
        SimpleNamespace(forward_mode=_ForwardMode(False)),
        metadata,
    )
    assert indexer.selected == "prefill"
    assert indexer.logical_positions.tolist() == [0]
    assert prefill_result.item() == 1

    positions.fill_(99)
    decode_result = QSAIndexer.forward_cuda(
        indexer,
        inputs,
        positions,
        SimpleNamespace(forward_mode=_ForwardMode(True)),
        metadata,
    )
    assert indexer.selected == "decode"
    assert indexer.logical_positions.tolist() == [6]
    assert decode_result.item() == 2


def test_qsa_fast_topk_returns_sequence_relative_indices():
    logits = torch.zeros(2, 1200, dtype=torch.float32)
    starts = torch.tensor([100, 600], dtype=torch.int32)
    ends = torch.tensor([700, 1150], dtype=torch.int32)
    logits[0, 321] = 10
    logits[1, 987] = 11
    indices = qsa_fast_topk(logits, starts, ends, topk=BLOCK_TOPK)
    assert 321 - 100 in indices[0].tolist()
    assert 987 - 600 in indices[1].tolist()

    decode_starts = torch.zeros(2, dtype=torch.int32)
    decode_ends = torch.tensor([700, 550], dtype=torch.int32)
    logits[0, 444] = 12
    logits[1, 333] = 13
    decode_indices = qsa_fast_topk(logits, decode_starts, decode_ends, topk=BLOCK_TOPK)
    assert 444 in decode_indices[0].tolist()
    assert 333 in decode_indices[1].tolist()


def test_qsa_reranks_wider_candidate_set():
    from sglang.srt.layers.attention.qsa.kernel import (
        _rerank_qsa_topk_candidates,
    )

    logits = torch.tensor(
        [[0.0, 9.0, 8.0, 7.0, 6.0, 10.0]], dtype=torch.float32
    )
    starts = torch.tensor([1], dtype=torch.int32)
    candidates = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)

    actual = _rerank_qsa_topk_candidates(logits, candidates, starts, topk=2)
    assert actual.tolist() == [[4, 0]]


def test_qsa_block_expansion_adds_only_incomplete_tail():
    blocks = torch.full((2, BLOCK_TOPK), -1, dtype=torch.int32)
    blocks[0, 0] = 0
    blocks[1, :2] = torch.tensor([1, 0])
    result = expand_qsa_block_indices(
        blocks,
        query_positions=torch.tensor([5, 10]),
        sequence_lengths=torch.tensor([6, 11]),
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
    )
    assert result.shape == (2, FINAL_TOPK)
    assert result[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert sorted(result[1, :11].tolist()) == list(range(11))
    assert torch.all(result[0, 6:] == -1)
    assert torch.all(result[1, 11:] == -1)


def test_qsa_triton_block_expansion_matches_torch_reference():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    # Regression coverage for the largest default CUDA graph batch.  The old
    # 2-D Triton expansion exceeded the per-program element limit at bs=512.
    rows = 512
    blocks = torch.full((rows, BLOCK_TOPK), -1, dtype=torch.int32)
    query_positions = torch.randint(0, 3000, (rows,), dtype=torch.int64)
    sequence_lengths = query_positions + 1
    complete_blocks = (query_positions + 1) // COMPRESS_RATIO
    valid_counts = torch.randint(0, BLOCK_TOPK + 1, (rows,))
    for row in range(rows):
        width = min(int(valid_counts[row]), int(complete_blocks[row]))
        if width:
            blocks[row, :width] = torch.randperm(int(complete_blocks[row]))[:width].to(
                torch.int32
            )

    expected = torch_expand_qsa_block_indices(
        blocks,
        query_positions,
        sequence_lengths,
        COMPRESS_RATIO,
        TOKEN_TOPK,
    )
    actual = triton_expand_qsa_block_indices(
        blocks.cuda(),
        query_positions.cuda(),
        sequence_lengths.cuda(),
        COMPRESS_RATIO,
        TOKEN_TOPK,
    ).cpu()
    assert torch.equal(actual, expected)


def test_qsa_decode_mqa_reads_paged_cache():
    torch.manual_seed(7)
    q = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    cache = torch.randn(8, 64, 1, 128, dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1, 5], [4, 2, 0]], dtype=torch.int32)
    context_lens = torch.tensor([130, 67], dtype=torch.int32)
    actual = qsa_mqa_decode(
        q,
        cache,
        page_table,
        context_lens,
        max_model_len=192,
    )

    gathered = cache[page_table.long(), :, 0].reshape(2, 192, 128)
    expected = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    expected = torch.relu(expected).sum(-1) / (128**0.5)
    positions = torch.arange(192).unsqueeze(0)
    expected.masked_fill_(positions >= context_lens[:, None], -float("inf"))
    torch.testing.assert_close(actual, expected)


def test_qsa_sparse_attention_matches_explicit_gqa():
    torch.manual_seed(2)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    v = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    slots = torch.full((2, FINAL_TOPK), -1, dtype=torch.int32)
    slots[0, :4] = torch.tensor([0, 2, 4, 6])
    slots[1, :3] = torch.tensor([1, 3, 5])
    actual = qsa_sparse_attention(q, k, v, slots)
    expected_rows = []
    for row, selected in enumerate((slots[0, :4], slots[1, :3])):
        keys = k[selected.long()].repeat_interleave(2, dim=1)
        values = v[selected.long()].repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) / 4
        expected_rows.append(
            torch.einsum("hk,khd->hd", scores.softmax(-1), values.float()).to(
                torch.bfloat16
            )
        )
    torch.testing.assert_close(actual, torch.stack(expected_rows), rtol=2e-2, atol=2e-2)


def test_qsa_mtp_step_out_cache_loc_matches_draft_forward_layout():
    """Each MTP draft step's metadata must reference the same out_cache_loc
    slice EAGLEWorker.draft_forward assigns for that step. The failure mode
    this guards: every QSA step writing its key state into the first
    ``batch_size`` slots of the shared buffer (the per-step layout tests
    that covered this went with the loc-map suite; the helper is unchanged
    and still the only mapping between the two layouts)."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseMultiStepDraftBackend,
    )

    backend = QwenSparseMultiStepDraftBackend.__new__(QwenSparseMultiStepDraftBackend)
    backend.topk, backend.speculative_num_steps = 1, 3
    bs, topk, steps = 4, 1, 3
    flat = torch.arange(bs * topk * steps, dtype=torch.int64)
    fb = SimpleNamespace(
        out_cache_loc=flat, batch_size=bs, seq_lens=torch.ones(bs)
    )
    # Reference: the exact draft_forward expression chain.
    reference = flat.reshape(bs, topk, steps).permute(2, 0, 1).reshape(steps, -1)
    for step in range(steps):
        got = backend._step_out_cache_loc(fb, step)
        assert torch.equal(got, reference[step]), step
    # Steps must address disjoint slots -- the guarded regression is every
    # step landing on the first bs slots.
    step_slices = [backend._step_out_cache_loc(fb, i) for i in range(steps)]
    all_slots = torch.cat(step_slices)
    assert all_slots.unique().numel() == bs * topk * steps
    # Non-draft layouts pass through untouched instead of inventing slots.
    odd = SimpleNamespace(
        out_cache_loc=torch.arange(5), batch_size=bs, seq_lens=torch.ones(bs)
    )
    assert torch.equal(backend._step_out_cache_loc(odd, 1), odd.out_cache_loc)
    backend.speculative_num_steps = 1
    assert torch.equal(backend._step_out_cache_loc(fb, 0), flat)


def test_qsa_graph_metadata_kernels_match_legacy_host_path():
    """The recorded replay kernels and the legacy host refresh must produce
    identical graph buffers from the same lengths + ledger sidecar, for both
    decode rows and target-verify row fan-out (boundary and non-boundary)."""
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    device = "cuda"
    ratio, full_page = 4, 64

    class _Pool:
        qsa_compress_ratio = ratio
        # compressed slots per full-KV page (page 64 tokens / ratio 4)
        qsa_compressed_page_size = full_page // ratio
        qsa_block_topk = 512

    def run_case(mode, bs, num_rows, seq_lens_list, extend_len, extend_lens=None):
        pool = _Pool()
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        # A page-aligned token table: request r's token i lives in full page
        # (r * 32 + i // 64) at offset i % 64, mirroring the paged allocator.
        rows = torch.arange(8, dtype=torch.int32, device=device)[:, None]
        cols = torch.arange(2048, dtype=torch.int32, device=device)[None, :]
        backend.req_to_token = (
            (rows * 32 + cols // full_page) * full_page + cols % full_page
        ).contiguous()
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)

        def make_metadata():
            indexer = QSAIndexerMetadata(
                sequence_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                token_to_batch_idx=torch.arange(
                    num_rows, dtype=torch.int32, device=device
                ),
                token_slot_table=torch.zeros(
                    (num_rows, 1), dtype=torch.int32, device=device
                ),
                out_cache_loc=torch.zeros(num_rows, dtype=torch.int64, device=device),
                token_to_kv_pool=pool,
                compress_ratio=ratio,
                block_topk=512,
                req_pool_indices=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                is_cuda_graph=True,
                graph_write_locs=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                graph_compressed_page_table=torch.zeros(
                    (num_rows, 32), dtype=torch.int32, device=device
                ),
                graph_compressed_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                graph_prefix_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                decode_logical_positions=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                pending_ring_slots=torch.zeros(
                    num_rows, dtype=torch.int64, device=device
                ),
                graph_ring_group_locs=torch.zeros(
                    (num_rows, ratio), dtype=torch.int32, device=device
                ),
            )
            return qsa_backend_module.QwenSparseAttnMetadata(
                sequence_lengths=indexer.sequence_lengths,
                token_to_batch_idx=indexer.token_to_batch_idx,
                token_slot_table=indexer.token_slot_table,
                indexer_metadata=indexer,
                row_req_pool_indices=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                is_cuda_graph=True,
            )

        # Path 1: recorded kernels.
        kernel_metadata = make_metadata()
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_lens=(
                None
                if extend_lens is None
                else torch.tensor(extend_lens, dtype=torch.int32, device=device)
            ),
            extend_len=extend_len,
            num_padding=0,
            metadata=kernel_metadata,
            req_to_token=backend.req_to_token,
            pool=pool,
        )
        # Path 2: legacy host refresh over the layout the kernels produced.
        host_metadata = make_metadata()
        host_metadata.sequence_lengths.copy_(kernel_metadata.sequence_lengths)
        host_metadata.row_req_pool_indices.copy_(
            kernel_metadata.row_req_pool_indices
        )
        backend._update_qsa_cuda_graph_metadata(
            host_metadata.indexer_metadata, host_metadata.row_req_pool_indices
        )
        for field in (
            "graph_write_locs",
            "graph_compressed_page_table",
            "graph_compressed_lengths",
            "decode_logical_positions",
            "pending_ring_slots",
            "graph_ring_group_locs",
        ):
            kernel_buf = getattr(kernel_metadata.indexer_metadata, field)
            host_buf = getattr(host_metadata.indexer_metadata, field)
            assert torch.equal(kernel_buf, host_buf), (mode, field)

    # Decode: lengths straddling boundaries (256 is a boundary, others not).
    run_case(mode=0, bs=4, num_rows=4, seq_lens_list=[255, 256, 257, 512], extend_len=0)
    # Target verify: 2 requests x 3 draft rows, one request crossing a boundary.
    run_case(mode=1, bs=2, num_rows=6, seq_lens_list=[254, 300], extend_len=3)
    # Draft extend: ragged per-request rows plus a dummy-tail row.
    run_case(
        mode=2,
        bs=2,
        num_rows=6,
        seq_lens_list=[256, 303],
        extend_len=0,
        extend_lens=[3, 2],
    )


def _qsa_expected_graph_layout(
    *, mode, bs, num_rows, seq_lens, req_pool, extend_lens, extend_len, num_padding
):
    """Host reimplementation of the graph layout contract (per-request rows +
    padded dummy tail), used to check the kernels independently rather than
    against another implementation of the same walk."""
    real_reqs = bs - num_padding
    row_lens, row_prefix, row_reqs = [], [], []
    for pid in range(bs):
        base = seq_lens[pid]
        if mode == 0:
            row_lens.append(base)
            row_prefix.append(max(base - 1, 0))
            row_reqs.append(req_pool[pid])
            continue
        eff = (extend_len if pid < real_reqs else 0) if mode == 1 else (
            extend_lens[pid] if pid < real_reqs else 0
        )
        prefix, limit = (base, base + eff) if mode == 1 else (max(base - eff, 0), base)
        for j in range(eff):
            row_lens.append(min(prefix + 1 + j, limit))
            row_prefix.append(prefix)
            row_reqs.append(req_pool[pid])
    # Dummy tail: length 1, prefix 0, aliased to request slot 0 (never
    # allocated, so its pending-ring rows are the inert store dump).
    while len(row_lens) < num_rows:
        row_lens.append(1)
        row_prefix.append(0)
        row_reqs.append(0)
    return row_lens, row_prefix, row_reqs


def test_qsa_graph_layout_covers_speculative_rows_and_padded_tail():
    """The recorded layout kernel must reproduce the per-step speculative row
    fan-out (uniform target-verify window and per-request draft-extend lengths)
    and the padded dummy-tail contract, and the row-metadata kernel must derive
    compressed lengths / boundary write slots / group state locs from those
    rows under arithmetic compressed addressing (write slot = boundary token's
    full slot // ratio; non-boundary rows keep the inert reserved slot 0).
    """
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    device = "cuda"
    ratio, full_page = 4, 64

    class _Pool:
        qsa_compress_ratio = ratio
        qsa_compressed_page_size = full_page // ratio
        qsa_block_topk = 512

    def run_case(mode, bs, num_rows, seq_lens, extend_lens, extend_len, num_padding):
        pool = _Pool()
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        rows = torch.arange(8, dtype=torch.int32, device=device)[:, None]
        cols = torch.arange(4096, dtype=torch.int32, device=device)[None, :]
        backend.req_to_token = (
            (rows * 64 + cols // full_page) * full_page + cols % full_page
        ).contiguous()
        req_pool = list(range(bs))
        max_pages = 64
        indexer = QSAIndexerMetadata(
            sequence_lengths=torch.zeros(num_rows, dtype=torch.int32, device=device),
            token_to_batch_idx=torch.arange(num_rows, dtype=torch.int32, device=device),
            token_slot_table=torch.zeros((num_rows, 1), dtype=torch.int32, device=device),
            out_cache_loc=torch.zeros(num_rows, dtype=torch.int64, device=device),
            token_to_kv_pool=pool,
            compress_ratio=ratio,
            block_topk=512,
            req_pool_indices=torch.zeros(num_rows, dtype=torch.int32, device=device),
            is_cuda_graph=True,
            graph_write_locs=torch.zeros(num_rows, dtype=torch.int32, device=device),
            graph_compressed_page_table=torch.zeros(
                (num_rows, max_pages), dtype=torch.int32, device=device
            ),
            graph_compressed_lengths=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            graph_prefix_lengths=torch.zeros(num_rows, dtype=torch.int32, device=device),
            decode_logical_positions=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            pending_ring_slots=torch.zeros(num_rows, dtype=torch.int64, device=device),
            graph_ring_group_locs=torch.zeros(
                (num_rows, ratio), dtype=torch.int32, device=device
            ),
        )
        metadata = qsa_backend_module.QwenSparseAttnMetadata(
            sequence_lengths=indexer.sequence_lengths,
            token_to_batch_idx=indexer.token_to_batch_idx,
            token_slot_table=indexer.token_slot_table,
            indexer_metadata=indexer,
            row_req_pool_indices=torch.zeros(num_rows, dtype=torch.int32, device=device),
            is_cuda_graph=True,
        )
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            req_pool_indices=torch.tensor(req_pool, dtype=torch.int32, device=device),
            extend_lens=(
                None
                if extend_lens is None
                else torch.tensor(extend_lens, dtype=torch.int32, device=device)
            ),
            extend_len=extend_len,
            num_padding=num_padding,
            metadata=metadata,
            req_to_token=backend.req_to_token,
            pool=pool,
        )
        torch.cuda.synchronize()

        exp_lens, exp_prefix, exp_reqs = _qsa_expected_graph_layout(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=seq_lens,
            req_pool=req_pool,
            extend_lens=extend_lens,
            extend_len=extend_len,
            num_padding=num_padding,
        )
        assert metadata.sequence_lengths.tolist() == exp_lens, (mode, "lengths")
        assert indexer.graph_prefix_lengths.tolist() == exp_prefix, (mode, "prefix")
        assert metadata.row_req_pool_indices.tolist() == exp_reqs, (mode, "req rows")

        r2t = backend.req_to_token.cpu()
        for row, (length, req) in enumerate(zip(exp_lens, exp_reqs)):
            assert int(indexer.graph_compressed_lengths[row]) == length // ratio
            if length > 0 and length % ratio == 0:
                expect = int(r2t[req, length - 1]) // ratio
            else:
                expect = 0  # non-boundary rows keep the inert reserved slot
            assert int(indexer.graph_write_locs[row]) == expect, (row, length)

    # Target verify: uniform 4-token window, 2 padded request slots.
    run_case(
        mode=1, bs=6, num_rows=32, seq_lens=[254, 255, 256, 300, 7, 7],
        extend_lens=None, extend_len=4, num_padding=2,
    )
    # Draft extend: per-request extend lengths (accept-count dependent), padded.
    run_case(
        mode=2, bs=5, num_rows=24, seq_lens=[260, 512, 257, 9, 9],
        extend_lens=[3, 1, 4, 0, 0], extend_len=0, num_padding=2,
    )
    # Decode with a padded tail (dummy rows alias request slot 0).
    run_case(
        mode=0, bs=4, num_rows=4, seq_lens=[256, 1024, 1025, 4096],
        extend_lens=None, extend_len=0, num_padding=0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
