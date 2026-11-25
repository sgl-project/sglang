from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import mqa_logits_fused_topk
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _metadata(*, rows=3, kv_rows=16384, method=TopkTransformMethod.RAGGED):
    ks = torch.zeros(rows, dtype=torch.int32)
    lengths = torch.arange(kv_rows - rows + 1, kv_rows + 1, dtype=torch.int32)
    return SimpleNamespace(
        get_indexer_kvcache_range=lambda: (ks, lengths),
        get_indexer_seq_len_cpu=lambda: torch.tensor([kv_rows], dtype=torch.int32),
        get_dsa_extend_len_cpu=lambda: [rows],
        topk_transform_method=method,
        force_unfused_topk=False,
        attn_metadata=SimpleNamespace(
            topk_indices_offset=torch.zeros(rows, dtype=torch.int32)
        ),
    )


def _indexer():
    return SimpleNamespace(
        n_heads=8,
        head_dim=128,
        # 132 selects the FP8 index-K layout, 68 the FP4 one.
        _HEAD_DIM_WITH_SF=132,
        index_topk=128,
        dsa_enable_prefill_cp=False,
        sm_count=1,
        mqa_logits_fused_topk_kv_sample=(64, 4),
        bf16_head_gate=True,
        layer_id=0,
    )


def _inputs(rows=3, kv_rows=16384):
    q = torch.empty((rows, 8, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((kv_rows, 128), dtype=torch.float8_e4m3fn)
    return (
        q,
        (k, torch.ones(kv_rows, dtype=torch.float32)),
        torch.ones(rows, 8, dtype=torch.float32),
    )


def test_halops_dispatch_uses_ragged_bounds_and_existing_output():
    q, kv, weights = _inputs()
    metadata = _metadata()
    indexer = _indexer()
    output = torch.full((3, 128), -7, dtype=torch.int32)
    calls = {}
    sentinel = object()

    def fake_halops(**kwargs):
        calls.update(kwargs)
        return sentinel

    with (
        patch.object(
            mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", fake_halops
        ),
        patch.object(mqa_logits_fused_topk, "should_use_fused", return_value=True),
    ):
        result = mqa_logits_fused_topk.mqa_logits_fused_topk(
            indexer,
            metadata,
            q=q,
            q_sf=None,
            kv=kv,
            weights=weights,
            out=output,
        )

    assert result is sentinel
    assert calls["q"] is q
    assert calls["q_sf"] is None
    assert calls["kv"][0] is kv[0]
    assert calls["kv"][1] is kv[1]
    assert calls["weights"] is weights
    assert calls["ks"] is metadata.get_indexer_kvcache_range()[0]
    assert calls["ke"] is metadata.get_indexer_kvcache_range()[1]
    assert calls["topk_indices"] is output
    assert calls["top_k"] == 128
    assert calls["ordered_bounds"] is True
    assert calls["topk_indices_offset"] is metadata.attn_metadata.topk_indices_offset


def test_halops_dispatch_keeps_paged_batches_on_baseline():
    q, kv, weights = _inputs()
    metadata = _metadata(method=TopkTransformMethod.PAGED)

    with patch.object(
        mqa_logits_fused_topk,
        "_halops_mqa_logits_fused_topk",
        side_effect=AssertionError("paged batches must not use HALOps"),
    ):
        result = mqa_logits_fused_topk.mqa_logits_fused_topk(
            _indexer(),
            metadata,
            q=q,
            q_sf=None,
            kv=kv,
            weights=weights,
        )

    assert result is None


def test_halops_dispatch_keeps_forced_unfused_batches_on_baseline():
    q, kv, weights = _inputs()
    metadata = _metadata()
    metadata.force_unfused_topk = True

    with patch.object(
        mqa_logits_fused_topk,
        "_halops_mqa_logits_fused_topk",
        side_effect=AssertionError("forced-unfused batches must not use HALOps"),
    ):
        assert (
            mqa_logits_fused_topk.mqa_logits_fused_topk(
                _indexer(), metadata, q=q, q_sf=None, kv=kv, weights=weights
            )
            is None
        )


@pytest.mark.parametrize(
    "indexer_options,fuse_topk",
    [
        ({"num_init_tokens": 4}, True),
        ({"num_local_tokens": 4}, True),
        ({}, False),
    ],
)
def test_gate_rejects_indexers_the_kernel_cannot_serve(indexer_options, fuse_topk):
    indexer = _indexer()
    indexer.__dict__.update(indexer_options)
    warmup = Mock()

    with (
        envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.override(True),
        envs.SGLANG_DSA_FUSE_TOPK.override(fuse_topk),
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "_halops_warmup", warmup),
        patch.object(
            mqa_logits_fused_topk, "unsupported_reason", Mock(return_value=None)
        ),
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        assert not mqa_logits_fused_topk.resolve_indexer_gate(indexer)

    warmup.assert_not_called()


@pytest.mark.parametrize("use_fp4", [False, True])
@pytest.mark.parametrize("bf16_head_gate", [False, True])
def test_warmup_matches_runtime_quantization_and_fp32_score_weights(
    use_fp4, bf16_head_gate
):
    indexer = _indexer()
    indexer._HEAD_DIM_WITH_SF = 68 if use_fp4 else 132
    indexer.bf16_head_gate = bf16_head_gate
    warmup = Mock()
    supported = Mock(return_value=None)
    with (
        envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.override(True),
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "_halops_warmup", warmup),
        patch.object(mqa_logits_fused_topk, "unsupported_reason", supported),
        patch.object(mqa_logits_fused_topk, "resolve_kv_sample", return_value=(64, 4)),
        patch.object(mqa_logits_fused_topk, "_failed_warmups", set()),
        patch.object(mqa_logits_fused_topk, "_warmed", set()),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
    ):
        assert mqa_logits_fused_topk.resolve_indexer_gate(indexer)

    dtype = "mxfp4" if use_fp4 else "fp8"
    assert supported.call_args.kwargs["qkv_dtype"] == dtype
    warmup.assert_called_once_with(
        qkv_dtype=dtype,
        scores_bf16=False,
        num_heads=8,
        top_k=128,
        num_sms=1,
        kv_sample=(64, 4),
        ordered_bounds=True,
    )


def test_flashindexer_selection_narrows_the_kernel_head_width():
    """--enable-flashindexer keeps 8 heads per token, so the gate warms and the
    dispatch sizes the kernel for 8 heads although the indexer has 64."""
    from sglang.srt.layers.attention.dsa.flashindexer import FLASHINDEXER_TOPK_HEADS

    indexer = _indexer()
    indexer.n_heads = 64
    indexer.linear_gate = Mock()
    warmup = Mock()
    supported = Mock(return_value=None)
    with (
        envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.override(True),
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "_halops_warmup", warmup),
        patch.object(mqa_logits_fused_topk, "unsupported_reason", supported),
        patch.object(mqa_logits_fused_topk, "resolve_kv_sample", return_value=(64, 4)),
        patch.object(mqa_logits_fused_topk, "_failed_warmups", set()),
        patch.object(mqa_logits_fused_topk, "_warmed", set()),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
    ):
        assert mqa_logits_fused_topk.resolve_indexer_gate(indexer)

    assert supported.call_args.kwargs["qkv_dtype"] == "fp8"
    assert warmup.call_args.kwargs["num_heads"] == FLASHINDEXER_TOPK_HEADS

    # flashindexer_select_heads already cut q and weights down to the selected heads.
    q = torch.empty((3, FLASHINDEXER_TOPK_HEADS, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((16384, 128), dtype=torch.float8_e4m3fn)
    gate_calls = {}

    def fake_should_use_fused(*args, **kwargs):
        gate_calls.update(kwargs)
        return True

    with (
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "should_use_fused", fake_should_use_fused),
    ):
        mqa_logits_fused_topk.mqa_logits_fused_topk(
            indexer,
            _metadata(),
            q=q,
            q_sf=None,
            kv=(k, torch.ones(16384, dtype=torch.float32)),
            weights=torch.ones((3, FLASHINDEXER_TOPK_HEADS), dtype=torch.float32),
        )

    assert gate_calls["num_heads"] == FLASHINDEXER_TOPK_HEADS


@pytest.mark.parametrize("eligible", [False, True])
def test_fp4_dispatch_reinterprets_gathered_cache_and_quantizes_only_if_used(eligible):
    indexer = _indexer()
    indexer._HEAD_DIM_WITH_SF = 68
    q = torch.empty((3, 8, 128), dtype=torch.bfloat16)
    q_fp4 = torch.empty((3, 8, 64), dtype=torch.uint8)
    q_sf = torch.full((3, 8), 0x7F7F7F7F, dtype=torch.int32)
    k = torch.empty((16384, 64), dtype=torch.float8_e4m3fn)
    k_sf = torch.full((16384,), 0x7F7F7F7F, dtype=torch.int32)
    indexer._quantize_fp4 = Mock(return_value=(q_fp4, q_sf))
    output = torch.full((3, 128), -7, dtype=torch.int32)
    halops = Mock(return_value=output)
    with (
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", halops),
        patch.object(mqa_logits_fused_topk, "should_use_fused", return_value=eligible),
    ):
        result = mqa_logits_fused_topk.mqa_logits_fused_topk(
            indexer,
            _metadata(),
            q=q,
            q_sf=None,
            kv=(k, k_sf.view(torch.float32)),
            weights=torch.ones((3, 8), dtype=torch.float32),
            out=output,
        )

    if not eligible:
        assert result is None
        indexer._quantize_fp4.assert_not_called()
        halops.assert_not_called()
        return
    assert result is output
    indexer._quantize_fp4.assert_called_once_with(q)
    arguments = halops.call_args.kwargs
    assert arguments["q"] is q_fp4
    assert arguments["q_sf"] is q_sf
    assert arguments["kv"][0].dtype == torch.uint8
    assert arguments["kv"][0].data_ptr() == k.data_ptr()
    torch.testing.assert_close(arguments["kv"][1], k_sf)
    assert arguments["kv"][1].data_ptr() == k_sf.data_ptr()


def test_warmup_is_cached_per_kernel_config():
    indexer = _indexer()
    fp4_indexer = _indexer()
    fp4_indexer._HEAD_DIM_WITH_SF = 68
    warmup = Mock()
    with (
        envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.override(True),
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "_halops_warmup", warmup),
        patch.object(
            mqa_logits_fused_topk, "unsupported_reason", Mock(return_value=None)
        ),
        patch.object(mqa_logits_fused_topk, "resolve_kv_sample", return_value=(64, 4)),
        patch.object(mqa_logits_fused_topk, "_failed_warmups", set()),
        patch.object(mqa_logits_fused_topk, "_warmed", set()),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
    ):
        # Every layer resolves the same gate; only the first one warms the
        # kernel, and the fp4 layout is a separate config.
        assert mqa_logits_fused_topk.resolve_indexer_gate(indexer)
        assert mqa_logits_fused_topk.resolve_indexer_gate(indexer)
        assert warmup.call_count == 1
        assert mqa_logits_fused_topk.resolve_indexer_gate(fp4_indexer)
        assert warmup.call_count == 2


def test_failed_warmup_disables_the_gate_without_retrying():
    indexer = _indexer()
    warmup = Mock(side_effect=RuntimeError("no kernel for this config"))
    with (
        envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.override(True),
        patch.object(mqa_logits_fused_topk, "_halops_mqa_logits_fused_topk", Mock()),
        patch.object(mqa_logits_fused_topk, "_halops_warmup", warmup),
        patch.object(
            mqa_logits_fused_topk, "unsupported_reason", Mock(return_value=None)
        ),
        patch.object(mqa_logits_fused_topk, "resolve_kv_sample", return_value=(64, 4)),
        patch.object(mqa_logits_fused_topk, "_failed_warmups", set()),
        patch.object(mqa_logits_fused_topk, "_warmed", set()),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
    ):
        assert not mqa_logits_fused_topk.resolve_indexer_gate(indexer)
        assert not mqa_logits_fused_topk.resolve_indexer_gate(indexer)
        assert warmup.call_count == 1


@pytest.mark.parametrize(
    "cp_size,cp_rank,rows,expected",
    [(1, 0, 5, (34, 38)), (2, 0, 3, (22, 24)), (2, 1, 2, (12, 14))],
)
def test_work_estimate_handles_packed_requests_and_cp(cp_size, cp_rank, rows, expected):
    assert (
        mqa_logits_fused_topk._estimate_ragged_work(
            torch.tensor([10, 4], dtype=torch.int32),
            [3, 2],
            rows,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        == expected
    )


def test_work_estimate_rejects_mismatched_packed_rows():
    assert (
        mqa_logits_fused_topk._estimate_ragged_work(
            torch.tensor([10, 4], dtype=torch.int32),
            [3, 2],
            4,
        )
        is None
    )
