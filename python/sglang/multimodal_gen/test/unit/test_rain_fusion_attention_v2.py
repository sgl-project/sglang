# SPDX-License-Identifier: Apache-2.0
"""CPU forwarding contracts for the public MindIE-SD attention entry points.

The vendor mocks exercise adapter dispatch and sequence isolation. They do not
establish native support or numerical accuracy for any quantization mode.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn_v2 import (
    RainFusionAttentionV2MetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn_v2"


def _reference(query, key, value, **options):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=options["scale"],
    ).transpose(1, 2)


@pytest.fixture
def mindie(monkeypatch):
    vendor = ModuleType("mindiesd")
    vendor.sparse_attention = Mock(side_effect=_reference)
    vendor.quant_attention = Mock(side_effect=_reference)
    monkeypatch.setitem(sys.modules, "mindiesd", vendor)
    module = importlib.import_module(MODULE)
    return module, vendor.sparse_attention, vendor.quant_attention


def _impl(mindie, **options):
    config = dict(num_heads=2, head_size=128, causal=False, softmax_scale=0.37)
    config.update(options)
    return mindie[0].RainFusionAttentionV2Impl(**config)


def _metadata(**options):
    config = dict(
        current_timestep=10,
        video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}],
    )
    config.update(options)
    return RainFusionAttentionV2MetadataBuilder().build(**config)


def _context(monkeypatch, metadata):
    from sglang.multimodal_gen.runtime.managers import forward_context

    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )


@pytest.mark.parametrize("sparse_type", [None, "rf_v2", "rf_v3", "ada_bsa"])
@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_public_sparse_choices_are_forwarded(mindie, sparse_type, precision):
    query = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)

    output = _impl(mindie).forward(
        query, query, query, _metadata(sparse_type=sparse_type, precision=precision)
    )

    assert output.shape == query.shape
    mindie[2].assert_not_called()
    mindie[1].assert_called_once()
    options = mindie[1].call_args.kwargs
    assert options["sparse_type"] == sparse_type
    assert options["precision"] == precision
    assert options["input_layout"] == "BSND"
    assert options["head_num"] == 2
    assert options["scale"] == 0.37
    assert options["block_size"] == (
        256 if sparse_type == "rf_v3" and precision in ("fp8", "mxfp4") else 128
    )
    if sparse_type not in ("rf_v2", "rf_v3"):
        assert (
            not {"video_spans", "txt_len", "latent_shape_q", "latent_shape_k"}
            & options.keys()
        )


@pytest.mark.parametrize("precision, block_size", [("mix", 256), ("fp8", 128)])
def test_sparse_public_options_and_explicit_block_size_are_preserved(
    mindie, precision, block_size
):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    mask = torch.zeros(128, 128, dtype=torch.bool)

    _impl(mindie).forward(
        query,
        query,
        query,
        _metadata(
            precision=precision,
            block_size=block_size,
            inner_precise=2,
            sparsity=0.6,
            attn_mask=mask,
            keep_sink=False,
            keep_recent=False,
            cdf_threshold=0.7,
            input_layout="BNSD",
            head_num=99,
            scale=99.0,
        ),
    )

    options = mindie[1].call_args.kwargs
    assert options["block_size"] == block_size
    assert options["precision"] == precision
    assert options["inner_precise"] == 2
    assert options["sparsity"] == 0.6
    assert options["attn_mask"] is mask
    assert options["keep_sink"] is False
    assert options["keep_recent"] is False
    assert options["cdf_threshold"] == 0.7
    assert options["input_layout"] == "BSND"
    assert options["head_num"] == 2
    assert options["scale"] == 0.37


def test_sparse_causal_option_is_forwarded_to_the_vendor(mindie):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    _impl(mindie, causal=True).forward(query, query, query, _metadata())
    assert mindie[1].call_args.kwargs["is_causal"] is True


def test_native_grid_constraints_are_delegated_without_a_dense_retry(mindie):
    query = torch.zeros(1, 9, 2, 128, dtype=torch.bfloat16)
    metadata = _metadata(video_spans=[{"start": 0, "latent_shape": [1, 3, 3]}])
    _impl(mindie).forward(query, query, query, metadata)
    mindie[1].assert_called_once()
    mindie[2].assert_not_called()
    assert mindie[1].call_args.kwargs["video_spans"] == metadata.video_spans


@pytest.mark.parametrize("sparse_type", ["rf_v2", "rf_v3"])
def test_text_prefix_is_translated_to_a_protected_video_span(mindie, sparse_type):
    query = torch.zeros(1, 131, 2, 128, dtype=torch.bfloat16)
    metadata = _metadata(
        sparse_type=sparse_type,
        video_spans=None,
        raw_latent_shape=[1, 16, 1, 16, 32],
        patch_size=(1, 2, 2),
        txt_len=3,
    )

    _impl(mindie).forward(query, query, query, metadata)

    options = mindie[1].call_args.kwargs
    assert options["video_spans"] == [{"start": 3, "latent_shape": [1, 8, 16]}]
    assert not {"txt_len", "latent_shape_q", "latent_shape_k"} & options.keys()


def test_pure_video_uses_legacy_grid_arguments(mindie):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    metadata = _metadata(
        video_spans=None,
        raw_latent_shape=[1, 16, 1, 16, 32],
        patch_size=(1, 2, 2),
    )

    _impl(mindie).forward(query, query, query, metadata)

    options = mindie[1].call_args.kwargs
    assert options["latent_shape_q"] == [1, 8, 16]
    assert options["latent_shape_k"] == [1, 8, 16]
    assert "video_spans" not in options


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_dense_precision_selects_the_public_quantization_mode(mindie, precision):
    query = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)

    output = _impl(mindie).forward(
        query, query, query, _metadata(video_spans=[], dense_precision=precision)
    )

    assert output.shape == query.shape
    mindie[1].assert_not_called()
    mindie[2].assert_called_once()
    args, options = mindie[2].call_args
    assert all(tensor.shape == query.shape for tensor in args)
    assert all(tensor.is_contiguous() for tensor in args)
    assert options["precision"] == precision
    assert options["layout"] == "BSND"
    assert options["scale"] == 0.37
    if precision == "fp8":
        assert options["fp8_fa_mode"] == "HIGH_PRECISION"
    else:
        assert "fp8_fa_mode" not in options
    if precision == "mxfp4":
        assert options["layout_kv"] == options["layout_out"] == "BSND"


def test_dense_options_are_forwarded_without_overriding_adapter_layout(mindie):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    rotation = torch.eye(128, dtype=query.dtype)
    dense_options = dict(
        layout="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
        scale=99.0,
        precision="fp8",
        q_rot=rotation,
        k_rot=rotation,
        pre_tokens=17,
        next_tokens=19,
    )
    original = dense_options.copy()

    _impl(mindie).forward(
        query,
        query,
        query,
        _metadata(video_spans=[], dense_precision="mxfp4", dense_kwargs=dense_options),
    )

    options = mindie[2].call_args.kwargs
    assert options["layout"] == options["layout_kv"] == options["layout_out"] == "BSND"
    assert options["precision"] == "mxfp4"
    assert options["scale"] == 0.37
    assert options["q_rot"] is options["k_rot"] is rotation
    assert options["pre_tokens"] == 17 and options["next_tokens"] == 19
    assert dense_options == original


def test_explicit_fp8_mode_is_not_replaced_by_the_default(mindie):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    _impl(mindie).forward(
        query,
        query,
        query,
        _metadata(video_spans=[], dense_kwargs={"fp8_fa_mode": "C8V16_TILING512"}),
    )
    assert mindie[2].call_args.kwargs["fp8_fa_mode"] == "C8V16_TILING512"


def test_dense_tuple_result_is_unwrapped_for_the_framework(mindie):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    expected = torch.ones_like(query)
    operator = mindie[2]
    operator.side_effect = None
    operator.return_value = (expected, torch.zeros(1))

    output = _impl(mindie).forward(
        query,
        query,
        query,
        _metadata(
            video_spans=[],
            dense_kwargs={"return_softmax_lse": True},
        ),
    )

    torch.testing.assert_close(output, expected)
    assert operator.call_args.kwargs["return_softmax_lse"] is True


@pytest.mark.parametrize(
    "step, dense", [(0, True), (1, False), (2, False), (3, True), (4, True)]
)
def test_dense_schedule_boundaries_dispatch_inline(mindie, step, dense):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    _impl(mindie).forward(
        query,
        query,
        query,
        _metadata(
            current_timestep=step, skip_first_steps=1, dense_last_steps=2, num_steps=5
        ),
    )
    (mindie[2] if dense else mindie[1]).assert_called_once()
    (mindie[1] if dense else mindie[2]).assert_not_called()


@pytest.mark.parametrize("options", [{"sparsity": 0}, {"video_spans": []}])
def test_explicit_dense_inputs_use_quant_attention(mindie, options):
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    _impl(mindie).forward(query, query, query, _metadata(**options))
    mindie[1].assert_not_called()
    mindie[2].assert_called_once()


@pytest.mark.parametrize("dense", [False, True])
def test_batch_samples_have_independent_public_calls(mindie, dense):
    query = torch.zeros(2, 128, 2, 128, dtype=torch.bfloat16)
    value = torch.cat((torch.ones_like(query[:1]), torch.full_like(query[:1], 23.0)))

    output = _impl(mindie).forward(
        query, query, value, _metadata(current_timestep=0 if dense else 10)
    )

    torch.testing.assert_close(output, value)
    operator = mindie[2] if dense else mindie[1]
    assert operator.call_count == 2
    assert all(
        call.args[0].shape == (1, 128, 2, 128) for call in operator.call_args_list
    )
    (mindie[1] if dense else mindie[2]).assert_not_called()


def test_packed_documents_rebase_spans_and_preserve_isolation(mindie, monkeypatch):
    spans = [
        {"start": 3, "latent_shape": [1, 8, 16]},
        {"start": 136, "latent_shape": [1, 8, 16]},
    ]
    _context(monkeypatch, _metadata(video_spans=spans))
    query = torch.zeros(264, 2, 128, dtype=torch.bfloat16)
    value = torch.cat(
        (torch.ones_like(query[:131]), torch.full_like(query[131:], 23.0))
    )
    bounds = torch.tensor([0, 131, 264], dtype=torch.int32)
    monkeypatch.setattr(
        bounds, "tolist", Mock(side_effect=AssertionError("host bounds required"))
    )

    output = _impl(mindie).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=bounds,
        cu_seqlens_host=(0, 131, 264),
        max_seqlen=133,
    )

    torch.testing.assert_close(output, value)
    assert [call.args[0].shape[1] for call in mindie[1].call_args_list] == [131, 133]
    assert [
        call.kwargs["video_spans"][0]["start"] for call in mindie[1].call_args_list
    ] == [3, 5]
    assert spans[1]["start"] == 136
    mindie[2].assert_not_called()


def test_physical_padding_is_an_independent_dense_segment(mindie, monkeypatch):
    _context(monkeypatch, _metadata(video_spans=[]))
    query = torch.zeros(64, 2, 128, dtype=torch.bfloat16)
    value = torch.cat((torch.ones_like(query[:5]), torch.full_like(query[5:], 31.0)))

    output = _impl(mindie, packed_trailing_padding=True).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=torch.tensor([0, 5, 64]),
        max_seqlen=5,
    )

    torch.testing.assert_close(output, value)
    assert [call.args[0].shape[1] for call in mindie[2].call_args_list] == [5, 59]
    mindie[1].assert_not_called()


def test_empty_packed_segments_are_not_dispatched(mindie, monkeypatch):
    _context(monkeypatch, _metadata(video_spans=[]))
    query = torch.zeros(5, 2, 128, dtype=torch.bfloat16)
    output = _impl(mindie).forward_varlen(
        query,
        query,
        query,
        cu_seqlens=torch.tensor([0, 0, 5, 5]),
        max_seqlen=5,
    )
    torch.testing.assert_close(output, query)
    mindie[2].assert_called_once()


@pytest.mark.parametrize("dense", [False, True])
def test_partitioned_heads_and_fp16_are_passed_to_mindie(mindie, dense):
    query = torch.zeros(1, 128, 2, 64, dtype=torch.float16)
    output = _impl(mindie, num_heads=8, head_size=64).forward(
        query, query, query, _metadata(video_spans=[] if dense else None)
    )
    assert output.shape == query.shape and output.dtype == query.dtype
    operator = mindie[2] if dense else mindie[1]
    assert operator.call_args.args[0].shape[-2:] == (2, 64)
    if not dense:
        assert operator.call_args.kwargs["head_num"] == 2


@pytest.mark.parametrize("dense", [False, True])
def test_vendor_errors_are_propagated_without_retry(mindie, dense):
    operator = mindie[2] if dense else mindie[1]
    operator.side_effect = ValueError("MindIE native option error")
    query = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    metadata = _metadata(
        video_spans=[] if dense else None,
        **(
            {"dense_precision": "vendor-extension"}
            if dense
            else {"precision": "vendor-extension"}
        ),
    )
    with pytest.raises(ValueError, match="MindIE native option error"):
        _impl(mindie).forward(query, query, query, metadata)
    operator.assert_called_once()
    (mindie[1] if dense else mindie[2]).assert_not_called()
    assert operator.call_args.kwargs["precision"] == "vendor-extension"


def test_registry_and_packed_capability(mindie):
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    backend = mindie[0].RainFusionAttentionV2Backend
    assert backend.get_enum() is AttentionBackendEnum.RAIN_FUSION_ATTN_V2
    assert (
        backend.unsupported_requirements(AttentionRequirements(packed_varlen=True))
        == ()
    )
    assert (
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.RAIN_FUSION_ATTN_V2, 128, torch.bfloat16
        )
        == f"{MODULE}.RainFusionAttentionV2Backend"
    )
