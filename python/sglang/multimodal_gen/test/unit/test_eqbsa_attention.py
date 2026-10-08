# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch and layout contracts for MindIE-SD EQBSA v3."""

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
from sglang.multimodal_gen.runtime.layers.attention.backends.eqbsa_attn import (
    EQBSAAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.eqbsa_attn"


@pytest.fixture
def eqbsa(monkeypatch):
    mindie = ModuleType("mindiesd")
    mindie.sparse_attention = Mock(
        side_effect=lambda q, k, v, **kw: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=kw["scale"]
        ).transpose(1, 2)
    )
    monkeypatch.setitem(sys.modules, "mindiesd", mindie)
    module = importlib.import_module(MODULE)
    return module, mindie.sparse_attention


def impl(eqbsa, head_size=64, **kwargs):
    return eqbsa[0].EQBSAAttentionImpl(2, head_size, False, head_size**-0.5, **kwargs)


def meta(**kwargs):
    return EQBSAAttentionMetadataBuilder().build(current_timestep=10, **kwargs)


def context(monkeypatch, metadata):
    from sglang.multimodal_gen.runtime.managers import forward_context

    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )


@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_legacy_text_prefix_and_patch_grid(eqbsa, precision):
    q = torch.randn(2, 3 + 2 * 8 * 16, 2, 128)
    m = meta(
        raw_latent_shape=[2, 16, 2, 16, 32],
        patch_size=(1, 2, 2),
        txt_len=3,
        sparsity=0.6,
        precision=precision,
    )
    out = impl(eqbsa, head_size=128).forward(q, q, q, m)
    assert out.shape == q.shape
    assert eqbsa[1].call_args.kwargs == dict(
        input_layout="BSND",
        head_num=2,
        scale=128**-0.5,
        sparse_type="rf_v3",
        inner_precise=4,
        block_size=128,
        sparsity=0.6,
        precision=precision,
        txt_len=3,
        latent_shape_q=[2, 8, 16],
        latent_shape_k=[2, 8, 16],
    )


@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_multispan_offsets_and_no_legacy_arguments(eqbsa, precision):
    # This verifies public API forwarding, not native support for each mode.
    spans = [
        {"start": 5, "latent_shape": [2, 8, 8]},
        {"start": 140, "latent_shape": [1, 8, 16]},
    ]
    q = torch.randn(1, 280, 2, 128)
    impl(eqbsa, head_size=128).forward(
        q, q, q, meta(video_spans=spans, precision=precision)
    )
    options = eqbsa[1].call_args.kwargs
    assert options["video_spans"] == spans
    assert not {"txt_len", "latent_shape_q", "latent_shape_k"} & options.keys()
    assert options["sparse_type"] == "rf_v3"
    assert options["precision"] == precision


@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_warmup_has_no_sparse_layout_options(eqbsa, precision):
    q = torch.randn(1, 256, 2, 64)
    m = EQBSAAttentionMetadataBuilder().build(
        current_timestep=9,
        video_spans=[{"start": 0, "latent_shape": [2, 8, 16]}],
        precision=precision,
    )
    impl(eqbsa).forward(q, q, q, m)
    assert eqbsa[1].call_args.kwargs == dict(
        input_layout="BSND", head_num=2, scale=0.125, sparse_type=None
    )


def test_zero_sparsity_still_calls_v3(eqbsa):
    q = torch.randn(1, 128, 2, 64)
    impl(eqbsa).forward(
        q,
        q,
        q,
        meta(video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}], sparsity=0),
    )
    assert eqbsa[1].call_args.kwargs["sparse_type"] == "rf_v3"


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_padding_is_isolated_and_host_bounds_avoid_sync(eqbsa, monkeypatch, precision):
    spans = [{"start": 4, "latent_shape": [2, 8, 8]}]
    context(monkeypatch, meta(video_spans=spans, precision=precision))
    q = torch.zeros(137, 2, 128)
    v = torch.cat([torch.ones(132, 2, 128), torch.full((5, 2, 128), 10000.0)])
    bounds = Mock()
    bounds.tolist.side_effect = AssertionError("host bounds required")
    out = impl(eqbsa, head_size=128).forward_varlen(
        q, q, v, cu_seqlens=bounds, cu_seqlens_host=(0, 132, 137), max_seqlen=132
    )
    torch.testing.assert_close(out[:132], torch.ones_like(out[:132]))
    torch.testing.assert_close(out[132:], v[132:])
    assert eqbsa[1].call_args_list[0].kwargs["precision"] == precision
    assert "precision" not in eqbsa[1].call_args_list[1].kwargs
    assert [call.args[0].shape[1] for call in eqbsa[1].call_args_list] == [132, 5]
    assert [call.kwargs["sparse_type"] for call in eqbsa[1].call_args_list] == [
        "rf_v3",
        None,
    ]


def test_two_video_documents_rebase_spans_independently(eqbsa, monkeypatch):
    spans = [
        {"start": 3, "latent_shape": [1, 8, 16]},
        {"start": 136, "latent_shape": [1, 8, 16]},
    ]
    context(monkeypatch, meta(video_spans=spans))
    q = torch.zeros(264, 2, 64)
    v = torch.cat([torch.ones(131, 2, 64), torch.full((133, 2, 64), 23.0)])
    out = impl(eqbsa).forward_varlen(
        q, q, v, cu_seqlens=torch.tensor([0, 131, 264]), max_seqlen=133
    )
    torch.testing.assert_close(out[:131], v[:131])
    torch.testing.assert_close(out[131:], v[131:])
    assert [c.kwargs["video_spans"][0]["start"] for c in eqbsa[1].call_args_list] == [
        3,
        5,
    ]


@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_text_refiner_does_not_use_joint_video_context(eqbsa, monkeypatch, precision):
    context(
        monkeypatch,
        meta(
            video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}], precision=precision
        ),
    )
    q = torch.randn(5, 2, 64)
    impl(eqbsa, prefix="token_refiner.blocks.0.attn").forward_varlen(
        q, q, q, cu_seqlens=torch.tensor([0, 5, 5]), max_seqlen=5
    )
    assert eqbsa[1].call_args.kwargs == dict(
        input_layout="BSND", head_num=2, scale=0.125, sparse_type=None
    )


def test_span_crossing_document_fails_before_any_dispatch(eqbsa, monkeypatch):
    context(monkeypatch, meta(video_spans=[{"start": 4, "latent_shape": [1, 8, 16]}]))
    q = torch.randn(150, 2, 64)
    with pytest.raises(ValueError, match="cross packed"):
        impl(eqbsa).forward_varlen(
            q, q, q, cu_seqlens=torch.tensor([0, 100, 150]), max_seqlen=100
        )
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("bounds", [(1, 5), (0, 4), (0, 6, 5), (0,)])
def test_invalid_packed_bounds(eqbsa, bounds):
    q = torch.randn(5, 2, 64)
    with pytest.raises(ValueError, match="monotonically cover"):
        impl(eqbsa).forward_varlen(
            q, q, q, cu_seqlens=torch.tensor(bounds), max_seqlen=5
        )
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": -1, "latent_shape": [1, 8, 8]}],
        [{"start": 0, "latent_shape": [1, 0, 8]}],
        [
            {"start": 0, "latent_shape": [1, 8, 8]},
            {"start": 63, "latent_shape": [1, 8, 8]},
        ],
        [{"start": 0.5, "latent_shape": [1, 8, 8]}],
    ],
)
def test_invalid_spans_rejected(spans):
    with pytest.raises(ValueError):
        meta(video_spans=spans)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparsity": 1},
        {"sparsity": -0.1},
        {"sparsity": float("nan")},
        {"skip_first_steps": -1},
        {"txt_len": True},
        {"precision": "float"},
        {"precision": "mxfp8"},
        {"precision": None},
    ],
)
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        meta(video_spans=[], **kwargs)


def test_incompatible_layout_arguments():
    with pytest.raises(ValueError, match="cannot be combined"):
        meta(video_spans=[], txt_len=1)
    with pytest.raises(ValueError, match="cannot be combined"):
        meta(video_spans=[], raw_latent_shape=[1, 16, 16], patch_size=(1, 2, 2))
    with pytest.raises(ValueError, match="divisible"):
        meta(raw_latent_shape=[1, 17, 16], patch_size=(1, 2, 2))


def test_sequence_mismatch_not_inferred_as_text(eqbsa):
    q = torch.randn(1, 130, 2, 64)
    with pytest.raises(ValueError, match="Legacy EQBSA layout"):
        impl(eqbsa).forward(
            q, q, q, meta(raw_latent_shape=[1, 8, 16], patch_size=(1, 1, 1))
        )
    eqbsa[1].assert_not_called()


def test_too_few_dense_fillers_uses_dense(eqbsa):
    # Valid multi-frame grids isolate the filler guard from grid fallbacks.
    # The first 200-token span needs 56 context tokens; none are available.
    q = torch.randn(1, 400, 2, 64)
    spans = [
        {"start": 0, "latent_shape": [2, 10, 10]},
        {"start": 200, "latent_shape": [2, 10, 10]},
    ]
    impl(eqbsa).forward(q, q, q, meta(video_spans=spans))
    assert eqbsa[1].call_args.kwargs["sparse_type"] is None


def test_native_errors_propagate(eqbsa):
    eqbsa[1].side_effect = RuntimeError("native EQBSA error")
    q = torch.randn(1, 128, 2, 64)
    with pytest.raises(RuntimeError, match="native EQBSA"):
        impl(eqbsa).forward(
            q, q, q, meta(video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}])
        )
    assert eqbsa[1].call_count == 1


def test_registry_and_packed_capability(eqbsa):
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    backend = eqbsa[0].EQBSAAttentionBackend
    assert backend.get_enum() is AttentionBackendEnum.EQBSA_ATTN
    assert AttentionBackendEnum.EQBSA_ATTN.is_sparse
    assert (
        backend.unsupported_requirements(AttentionRequirements(packed_varlen=True))
        == ()
    )
    assert not backend.supports_ring_rotation()
    assert (
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.EQBSA_ATTN, 128, torch.bfloat16
        )
        == f"{MODULE}.EQBSAAttentionBackend"
    )


def test_legacy_partial_text_block_uses_span_protection(eqbsa):
    q = torch.randn(1, 4 * 12 * 12 + 97, 2, 64)
    impl(eqbsa).forward(
        q, q, q, meta(raw_latent_shape=[4, 12, 12], patch_size=(1, 1, 1), txt_len=97)
    )
    options = eqbsa[1].call_args.kwargs
    assert options["video_spans"] == [{"start": 97, "latent_shape": [4, 12, 12]}]
    assert "txt_len" not in options
    assert options["sparse_type"] == "rf_v3"


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
def test_legacy_quantized_kv_text_boundary_uses_span_protection(eqbsa, precision):
    # Video ends on a Q=128 block boundary, but text spans two KV=256 blocks.
    # Passing legacy txt_len would protect only one KV block in MindIE.
    q = torch.randn(1, 384 + 129, 2, 128)
    impl(eqbsa, head_size=128).forward(
        q,
        q,
        q,
        meta(
            raw_latent_shape=[3, 8, 16],
            patch_size=(1, 1, 1),
            txt_len=129,
            precision=precision,
        ),
    )
    options = eqbsa[1].call_args.kwargs
    assert options["video_spans"] == [{"start": 129, "latent_shape": [3, 8, 16]}]
    assert not {"txt_len", "latent_shape_q", "latent_shape_k"} & options.keys()
    assert options["precision"] == precision
    assert options["sparse_type"] == "rf_v3"


@pytest.mark.parametrize("legacy", [False, True])
def test_unaligned_single_frame_uses_dense_without_rearrange(eqbsa, legacy):
    q = torch.randn(1, 80, 2, 64)
    kwargs = (
        dict(raw_latent_shape=[1, 8, 10], patch_size=(1, 1, 1))
        if legacy
        else dict(video_spans=[{"start": 0, "latent_shape": [1, 8, 10]}])
    )
    impl(eqbsa).forward(q, q, q, meta(**kwargs))
    assert eqbsa[1].call_args.kwargs["sparse_type"] is None


def test_missing_layout_rejected(eqbsa):
    q = torch.randn(1, 128, 2, 64)
    with pytest.raises(ValueError, match="explicit token layout"):
        impl(eqbsa).forward(q, q, q, None)
    eqbsa[1].assert_not_called()


def test_out_of_bounds_spans_rejected(eqbsa):
    q = torch.randn(1, 128, 2, 64)
    with pytest.raises(ValueError, match="within the sequence"):
        impl(eqbsa).forward(
            q, q, q, meta(video_spans=[{"start": 1, "latent_shape": [1, 8, 16]}])
        )
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{"causal": True}, {"num_kv_heads": 1}, {"dropout_p": 0.1}]
)
def test_unsupported_semantics_rejected(eqbsa, kwargs):
    options = dict(num_heads=2, head_size=64, causal=False, softmax_scale=0.125)
    options.update(kwargs)
    with pytest.raises(ValueError):
        eqbsa[0].EQBSAAttentionImpl(**options)
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("head_size", [32, 96])
def test_unsupported_native_head_sizes_fail_before_dispatch(eqbsa, head_size):
    with pytest.raises(ValueError, match="head sizes 64 and 128"):
        eqbsa[0].EQBSAAttentionImpl(2, head_size, False, 0.125)
    eqbsa[1].assert_not_called()


def test_mix_head_64_rejected_before_native_dispatch(eqbsa):
    q = torch.randn(1, 128, 2, 64)
    with pytest.raises(ValueError, match="head size 128"):
        impl(eqbsa).forward(
            q,
            q,
            q,
            meta(
                video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}], precision="mix"
            ),
        )
    eqbsa[1].assert_not_called()
