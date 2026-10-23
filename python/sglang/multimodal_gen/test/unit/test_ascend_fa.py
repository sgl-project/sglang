# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for explicit Ascend FA configuration; no native quality claims."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa


def implementation(config=None, **kwargs):
    options = dict(num_heads=2, head_size=64, causal=False, softmax_scale=0.125)
    options.update(kwargs)
    return ascend_fa.AscendFAImpl(attention_config=config, **options)


@pytest.fixture
def mindie(monkeypatch):
    module = ModuleType("mindiesd")

    def execute(q, k, v, **kwargs):
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k, v = (tensor.repeat_interleave(repeat, dim=1) for tensor in (k, v))
        return F.scaled_dot_product_attention(q, k, v, scale=kwargs["scale"]).transpose(
            1, 2
        )

    module.quant_attention = Mock(side_effect=execute)
    monkeypatch.setitem(sys.modules, "mindiesd", module)
    return module.quant_attention


@pytest.fixture
def native(monkeypatch):
    def execute(q, k, v, **kwargs):
        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k, v = (tensor.repeat_interleave(repeat, dim=1) for tensor in (k, v))
        mask = kwargs.get("atten_mask")
        result = F.scaled_dot_product_attention(
            q, k, v, scale=kwargs["scale"], attn_mask=None if mask is None else ~mask
        )
        return result, torch.zeros((*q.shape[:-1], 1), dtype=torch.float32)

    operator = Mock(side_effect=execute)
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score", operator, raising=False
    )
    return operator


def test_explicit_empty_configuration_is_bf16_and_copied():
    config = {}
    resolved = ascend_fa.validate_attention_config(config)
    assert resolved == {"precision": "bf16"} and config == {}
    adapter = implementation(config)
    config["precision"] = "fp8"
    assert adapter.attention_config == {"precision": "bf16"}


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {"precision": "mix"},
        {"precision": None},
        {"unknown": 1},
        {"precision": "fp8", "scale": 0.5},
        {"precision": "fp8", "layout": "BSND"},
        {"precision": "bf16", "fp8_fa_mode": "HIGH_PRECISION"},
        {"precision": "mxfp8", "fp8_fa_mode": "HIGH_PRECISION"},
        {"precision": "fp8", "fp8_fa_mode": False},
        {"precision": "fp8", "pre_tokens": 1.5},
        {"precision": "fp8", "q_rot": [[1.0]]},
        {"precision": "bf16", "enable_laser": 1},
        {"precision": "fp8", "enable_laser": True},
        {"precision": "mxfp4", "layout_out": "BNSD"},
        {"precision": "mxfp4", "metadata": {}},
        {"precision": "mxfp4", "return_softmax_lse": 1},
        {"precision": "mxfp4", "mask_mode": True},
        {"precision": "mxfp4", "mxfp4_dst_type_max": float("nan")},
    ],
)
def test_unknown_or_unsupported_dense_options_fail(config):
    with pytest.raises(ValueError):
        ascend_fa.validate_attention_config(config)


def test_no_configuration_preserves_checkpoint_offline_mxfp8(monkeypatch):
    resolver = Mock(return_value="MXFP8")
    rotation = Mock(
        side_effect=AssertionError("offline checkpoint must not rotate online")
    )
    monkeypatch.setattr(ascend_fa, "resolve_mx_fa_scheme", resolver)
    monkeypatch.setattr(ascend_fa.AscendFAImpl, "_ensure_rot_matrix", rotation)
    checkpoint = SimpleNamespace(use_offline_qk_rotation=True)
    adapter = implementation(quant_config=checkpoint)
    assert adapter.attention_config is None
    assert adapter._quant_scheme == "MXFP8" and adapter.use_offline_qk_rotation
    resolver.assert_called_once_with(checkpoint)
    rotation.assert_not_called()


@pytest.mark.parametrize("precision", ["bf16", "fp8", "mxfp8", "mxfp4"])
def test_explicit_configuration_overrides_checkpoint_selection(monkeypatch, precision):
    resolver = Mock(side_effect=AssertionError("explicit configuration must win"))
    monkeypatch.setattr(ascend_fa, "resolve_mx_fa_scheme", resolver)
    adapter = implementation({"precision": precision}, quant_config=object())
    assert adapter._quant_scheme is None
    resolver.assert_not_called()


def test_bf16_uses_native_fia_without_mindie(native, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    query = torch.zeros(2, 8, 2, 64, dtype=torch.bfloat16)
    result, lse = implementation({}).forward(
        query, query, query, None, return_softmax_lse=True
    )
    assert result.shape == query.shape and lse.shape == (2, 2, 8)
    assert native.call_args.kwargs["input_layout"] == "BNSD"
    assert native.call_args.kwargs["scale"] == 0.125
    assert native.call_args.kwargs["softmax_lse_flag"] is True


def test_bf16_causal_mask_is_preserved(native):
    query = torch.zeros(1, 4, 2, 64, dtype=torch.bfloat16)
    implementation({}, causal=True).forward(query, query, query, None)
    mask = native.call_args.kwargs["atten_mask"]
    assert torch.equal(
        mask[0], torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    )


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_quantized_samples_are_isolated_and_return_bsnd(mindie, precision):
    query = torch.zeros(2, 8, 2, 64, dtype=torch.bfloat16)
    value = torch.cat((torch.ones_like(query[:1]), torch.full_like(query[:1], 23)))
    result = implementation({"precision": precision}).forward(query, query, value, None)
    torch.testing.assert_close(result, value)
    assert mindie.call_count == 2
    for call in mindie.call_args_list:
        assert call.args[0].shape == (1, 8, 2, 64)
        assert all(tensor.is_contiguous() for tensor in call.args)
        assert call.kwargs["precision"] == precision
        assert call.kwargs["layout"] == "BSND" and call.kwargs["scale"] == 0.125
        if precision == "fp8":
            assert call.kwargs["fp8_fa_mode"] == "HIGH_PRECISION"
        if precision == "mxfp4":
            assert call.kwargs["layout_kv"] == call.kwargs["layout_out"] == "BSND"


def test_mxfp4_public_options_are_forwarded_without_layout_override(mindie):
    rotation = torch.eye(64, dtype=torch.bfloat16)
    native_metadata = torch.zeros(1, dtype=torch.int64)
    config = dict(
        precision="mxfp4",
        q_rot=rotation,
        k_rot=rotation,
        mxfp4_scale_alg=1,
        mxfp4_dst_type_max=6.0,
        metadata=native_metadata,
        num_key_value_heads=2,
        max_seqlen_q=256,
        max_seqlen_kv=512,
        mask_mode=0,
        win_left=8,
        win_right=0,
        return_softmax_lse=0,
    )
    query = torch.zeros(1, 16, 2, 64, dtype=torch.bfloat16)
    key = torch.zeros(1, 20, 2, 64, dtype=torch.bfloat16)
    result = implementation(config).forward(query, key, key, None)
    assert result.shape == query.shape
    options = mindie.call_args.kwargs
    for name, value in config.items():
        assert (
            options[name] is value
            if isinstance(value, torch.Tensor)
            else options[name] == value
        )
    assert options["layout"] == options["layout_kv"] == options["layout_out"] == "BSND"
    assert options["scale"] == 0.125


def test_quantized_head_partitions_and_gqa_are_not_mha_only(mindie):
    query = torch.zeros(1, 8, 4, 64, dtype=torch.bfloat16)
    key = torch.zeros(1, 8, 2, 64, dtype=torch.bfloat16)
    result = implementation(
        {"precision": "mxfp8"}, num_heads=8, num_kv_heads=4
    ).forward(query, key, key, None)
    assert result.shape == query.shape
    assert mindie.call_args.args[0].shape[2] == 4
    assert mindie.call_args.args[1].shape[2] == 2


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_quantized_causal_dropout_lse_and_ring_are_rejected(mindie, precision):
    config = {"precision": precision}
    with pytest.raises(NotImplementedError, match="causal"):
        implementation(config, causal=True)
    with pytest.raises(NotImplementedError, match="dropout"):
        implementation(config, dropout_p=0.1)
    adapter = implementation(config)
    query = torch.zeros(1, 8, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="LSE"):
        adapter.forward(query, query, query, None, return_softmax_lse=True)
    with pytest.raises(NotImplementedError, match="Ring/LSE"):
        adapter.forward_ring_kv_chunk(query[0], query[0], query[0])
    mindie.assert_not_called()


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_explicit_packed_documents_remain_independent(mindie, native, precision):
    query = torch.zeros(12, 2, 64, dtype=torch.bfloat16)
    value = torch.cat((torch.ones_like(query[:5]), torch.full_like(query[5:], 23)))
    result = implementation({"precision": precision}).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=torch.tensor([0, 5, 12]),
        cu_seqlens_host=(0, 5, 12),
        max_seqlen=7,
    )
    torch.testing.assert_close(result, value)
    operator = native if precision == "bf16" else mindie
    assert operator.call_count == 2


def test_explicit_h3_padding_is_zero_and_not_a_document(mindie):
    query = torch.zeros(8, 2, 64, dtype=torch.bfloat16)
    value = torch.cat((torch.ones_like(query[:5]), torch.full_like(query[5:], 500)))
    result = implementation(
        {"precision": "fp8"}, packed_trailing_padding=True
    ).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=torch.tensor([0, 5, 8]),
        cu_seqlens_host=(0, 5, 8),
        max_seqlen=5,
    )
    torch.testing.assert_close(result[:5], value[:5])
    assert not result[5:].any()
    mindie.assert_called_once()


@pytest.mark.parametrize(
    "host, maximum", [((0, 2.5, 5), 3), ((0, True, 5), 5), ((0, 5), True), ((0, 5), 4)]
)
def test_explicit_packed_metadata_is_strict(mindie, host, maximum):
    query = torch.zeros(5, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        implementation({"precision": "fp8"}).forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.tensor(host, dtype=torch.int32),
            cu_seqlens_host=host,
            max_seqlen=maximum,
        )
    mindie.assert_not_called()


@pytest.mark.parametrize(
    "shape, causal, lse, supported",
    [
        ((1, 2048, 2, 64), False, False, True),
        ((1, 2047, 2, 64), False, False, False),
        ((1, 2048, 2, 128), False, False, True),
        ((1, 2048, 2, 256), False, False, False),
        ((1, 2048, 2, 64), True, False, False),
        ((1, 2048, 2, 64), False, True, False),
    ],
)
def test_optional_laser_geometry_is_guarded(shape, causal, lse, supported):
    assert (
        ascend_fa._supports_laser_shape(
            shape, shape, shape, causal=causal, return_softmax_lse=lse
        )
        is supported
    )


def test_optional_laser_preserves_bf16_range_contract(monkeypatch):
    adapter = implementation({"enable_laser": True})
    laser = SimpleNamespace(
        _forward_dense=Mock(side_effect=lambda q, k, v, meta, **kw: v.clone())
    )
    monkeypatch.setattr(adapter, "_get_laser_impl", lambda *args: laser)
    query = torch.zeros(1, 8, 2, 64, dtype=torch.bfloat16)
    value = torch.ones_like(query)
    torch.testing.assert_close(adapter.forward(query, query, value, None), value)
    assert laser._forward_dense.call_args.kwargs == {"preserve_bf16_range": True}


def test_bf16_default_never_selects_laser(native, monkeypatch):
    adapter = implementation({})
    monkeypatch.setattr(
        adapter,
        "_get_laser_impl",
        Mock(side_effect=AssertionError("Laser requires explicit opt-in")),
    )
    query = torch.zeros(1, 2048, 2, 64, dtype=torch.bfloat16)
    adapter.forward(query, query, query, None)
    native.assert_called_once()


@pytest.mark.parametrize("name", ["attn_mask", "metadata", "block_table", "sinks"])
def test_batch_specific_mxfp4_tensors_reject_multibatch_and_packed_calls(mindie, name):
    config = {"precision": "mxfp4", name: torch.ones(1)}
    adapter = implementation(config)
    query = torch.zeros(2, 8, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="single sample"):
        adapter.forward(query, query, query, None)
    query = query.reshape(16, 2, 64)
    with pytest.raises(ValueError, match="one unpadded packed document"):
        adapter.forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 8, 16]), max_seqlen=8
        )
    mindie.assert_not_called()


@pytest.mark.parametrize("name", ["attn_mask", "metadata", "block_table", "sinks"])
def test_batch_config_guard_applies_before_external_document_splitting(name):
    config = {"precision": "mxfp4", name: torch.ones(1)}
    with pytest.raises(ValueError, match="one unpadded packed document"):
        ascend_fa.validate_attention_batch_config(
            config, bounds=(0, 8, 16), total_tokens=16
        )
    ascend_fa.validate_attention_batch_config(
        config, batch_size=1, bounds=(0, 16), total_tokens=16
    )


def test_shared_rotation_matrices_do_not_limit_packed_batches():
    ascend_fa.validate_attention_batch_config(
        {"precision": "fp8", "q_rot": torch.eye(64)},
        batch_size=2,
        bounds=(0, 8, 16),
        total_tokens=16,
    )


def test_quantized_native_errors_never_fall_back_to_bf16(mindie, native):
    mindie.side_effect = RuntimeError("native quantization failure")
    query = torch.zeros(1, 8, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="native quantization failure"):
        implementation({"precision": "fp8"}).forward(query, query, query, None)
    native.assert_not_called()


def test_missing_sdk_is_reported_without_native_fallback(native, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    query = torch.zeros(1, 8, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ImportError, match="requires MindIE-SD"):
        implementation({"precision": "fp8"}).forward(query, query, query, None)
    native.assert_not_called()
