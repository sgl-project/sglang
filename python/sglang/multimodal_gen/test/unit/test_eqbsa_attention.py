# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch and layout contracts for MindIE-SD EQBSA MIX."""

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
    EQBSAAttentionMetadata,
    EQBSAAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn_v2"


@pytest.fixture
def eqbsa(monkeypatch):
    mindie = ModuleType("mindiesd")
    operator = Mock(
        side_effect=lambda q, k, v, **kw: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=kw["scale"]
        ).transpose(1, 2)
    )

    # Expose the required public API parameters while recording only options
    # explicitly supplied by the adapter, including their absence in dense steps.
    def sparse_attention(q, k, v, *, precision=None, video_spans=None, **options):
        if precision is not None:
            options["precision"] = precision
        if video_spans is not None:
            options["video_spans"] = video_spans
        return operator(q, k, v, **options)

    mindie.sparse_attention = sparse_attention
    dense_operator = Mock(
        side_effect=lambda q, k, v, **kw: F.scaled_dot_product_attention(
            q, k, v, scale=kw["scale"]
        )
    )
    mindie.quant_attention = dense_operator
    monkeypatch.setitem(sys.modules, "mindiesd", mindie)
    module = importlib.import_module(MODULE)
    return module, operator, dense_operator


def impl(eqbsa, head_size=128, **kwargs):
    return eqbsa[0].EQBSAAttentionImpl(2, head_size, False, head_size**-0.5, **kwargs)


def meta(**kwargs):
    return EQBSAAttentionMetadataBuilder().build(
        current_timestep=kwargs.pop("current_timestep", 10), **kwargs
    )


def dense_options():
    return dict(
        precision="fp8",
        layout="BNSD",
        scale=128**-0.5,
        fp8_fa_mode="HIGH_PRECISION",
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )


def context(monkeypatch, metadata):
    from sglang.multimodal_gen.runtime.managers import forward_context

    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )


def test_legacy_text_prefix_and_patch_grid(eqbsa):
    q = torch.randn(2, 3 + 2 * 8 * 16, 2, 128, dtype=torch.bfloat16)
    m = meta(
        raw_latent_shape=[2, 16, 2, 16, 32],
        patch_size=(1, 2, 2),
        txt_len=3,
        sparsity=0.6,
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
        precision="mix",
        txt_len=3,
        latent_shape_q=[2, 8, 16],
        latent_shape_k=[2, 8, 16],
    )


def test_multispan_offsets_and_no_legacy_arguments(eqbsa):
    spans = [
        {"start": 5, "latent_shape": [2, 8, 8]},
        {"start": 140, "latent_shape": [1, 8, 16]},
    ]
    q = torch.randn(1, 280, 2, 128, dtype=torch.bfloat16)
    impl(eqbsa).forward(q, q, q, meta(video_spans=spans))
    options = eqbsa[1].call_args.kwargs
    assert options["video_spans"] == spans
    assert not {"txt_len", "latent_shape_q", "latent_shape_k"} & options.keys()
    assert options["sparse_type"] == "rf_v3"
    assert options["precision"] == "mix"


def test_warmup_has_no_sparse_layout_options(eqbsa):
    q = torch.randn(1, 256, 2, 128, dtype=torch.bfloat16)
    m = EQBSAAttentionMetadataBuilder().build(
        current_timestep=9,
        video_spans=[{"start": 0, "latent_shape": [2, 8, 16]}],
    )
    impl(eqbsa).forward(q, q, q, m)
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()
    assert eqbsa[2].call_args.args[0].shape == (1, 2, 256, 128)
    assert all(tensor.is_contiguous() for tensor in eqbsa[2].call_args.args)


def test_zero_sparsity_uses_fp8_fia(eqbsa):
    q = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)
    impl(eqbsa).forward(
        q,
        q,
        q,
        meta(video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}], sparsity=0),
    )
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()


def test_padding_is_isolated_and_host_bounds_avoid_sync(eqbsa, monkeypatch):
    spans = [{"start": 4, "latent_shape": [2, 8, 8]}]
    context(monkeypatch, meta(video_spans=spans))
    q = torch.zeros(137, 2, 128, dtype=torch.bfloat16)
    v = torch.cat(
        [
            torch.ones(132, 2, 128, dtype=q.dtype),
            torch.full((5, 2, 128), 10000.0, dtype=q.dtype),
        ]
    )
    bounds = torch.tensor([0, 132, 137], dtype=torch.int32)
    monkeypatch.setattr(
        bounds, "tolist", Mock(side_effect=AssertionError("host bounds required"))
    )
    out = impl(eqbsa, head_size=128).forward_varlen(
        q, q, v, cu_seqlens=bounds, cu_seqlens_host=(0, 132, 137), max_seqlen=132
    )
    torch.testing.assert_close(out[:132], torch.ones_like(out[:132]))
    torch.testing.assert_close(out[132:], v[132:])
    assert eqbsa[1].call_args_list[0].kwargs["precision"] == "mix"
    assert eqbsa[1].call_count == 1
    assert eqbsa[1].call_args.args[0].shape[1] == 132
    assert eqbsa[2].call_count == 1
    assert eqbsa[2].call_args.args[0].shape[2] == 5
    assert eqbsa[2].call_args.kwargs == dense_options()


def test_two_video_documents_rebase_spans_independently(eqbsa, monkeypatch):
    spans = [
        {"start": 3, "latent_shape": [1, 8, 16]},
        {"start": 136, "latent_shape": [1, 8, 16]},
    ]
    context(monkeypatch, meta(video_spans=spans))
    q = torch.zeros(264, 2, 128, dtype=torch.bfloat16)
    v = torch.cat(
        [
            torch.ones(131, 2, 128, dtype=q.dtype),
            torch.full((133, 2, 128), 23.0, dtype=q.dtype),
        ]
    )
    out = impl(eqbsa).forward_varlen(
        q, q, v, cu_seqlens=torch.tensor([0, 131, 264]), max_seqlen=133
    )
    torch.testing.assert_close(out[:131], v[:131])
    torch.testing.assert_close(out[131:], v[131:])
    assert [c.kwargs["video_spans"][0]["start"] for c in eqbsa[1].call_args_list] == [
        3,
        5,
    ]


def test_backend_prefix_does_not_bypass_metadata_contract(eqbsa):
    q = torch.randn(1, 5, 2, 128, dtype=torch.bfloat16)
    adapter = impl(eqbsa, prefix="token_refiner.blocks.0.attn")
    assert not hasattr(adapter, "text_refiner")
    with pytest.raises(ValueError, match="explicit token layout"):
        adapter.forward(q, q, q, None)
    eqbsa[1].assert_not_called()
    eqbsa[2].assert_not_called()


def test_span_crossing_document_fails_before_any_dispatch(eqbsa, monkeypatch):
    context(monkeypatch, meta(video_spans=[{"start": 4, "latent_shape": [1, 8, 16]}]))
    q = torch.randn(150, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cross packed"):
        impl(eqbsa).forward_varlen(
            q, q, q, cu_seqlens=torch.tensor([0, 100, 150]), max_seqlen=100
        )
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("bounds", [(1, 5), (0, 4), (0, 6, 5), (0,)])
def test_invalid_packed_bounds(eqbsa, bounds):
    q = torch.randn(5, 2, 128, dtype=torch.bfloat16)
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
    q = torch.randn(1, 130, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Legacy EQBSA layout"):
        impl(eqbsa).forward(
            q, q, q, meta(raw_latent_shape=[1, 8, 16], patch_size=(1, 1, 1))
        )
    eqbsa[1].assert_not_called()


def test_too_few_dense_fillers_uses_dense(eqbsa):
    # Valid multi-frame grids isolate the filler guard from grid fallbacks.
    # The first 200-token span needs 56 context tokens; none are available.
    q = torch.randn(1, 400, 2, 128, dtype=torch.bfloat16)
    spans = [
        {"start": 0, "latent_shape": [2, 10, 10]},
        {"start": 200, "latent_shape": [2, 10, 10]},
    ]
    impl(eqbsa).forward(q, q, q, meta(video_spans=spans))
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()


def test_native_errors_propagate(eqbsa):
    eqbsa[1].side_effect = RuntimeError("native EQBSA error")
    q = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="native EQBSA"):
        impl(eqbsa).forward(
            q, q, q, meta(video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}])
        )
    assert eqbsa[1].call_count == 1
    eqbsa[2].assert_not_called()


def test_registry_and_packed_capability(eqbsa):
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    backend = eqbsa[0].EQBSAAttentionBackend
    assert backend.get_supported_head_sizes() == [128]
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
    q = torch.randn(1, 4 * 12 * 12 + 97, 2, 128, dtype=torch.bfloat16)
    impl(eqbsa).forward(
        q, q, q, meta(raw_latent_shape=[4, 12, 12], patch_size=(1, 1, 1), txt_len=97)
    )
    options = eqbsa[1].call_args.kwargs
    assert options["video_spans"] == [{"start": 97, "latent_shape": [4, 12, 12]}]
    assert "txt_len" not in options
    assert options["sparse_type"] == "rf_v3"


@pytest.mark.parametrize("legacy", [False, True])
def test_unaligned_single_frame_uses_dense_without_rearrange(eqbsa, legacy):
    q = torch.randn(1, 80, 2, 128, dtype=torch.bfloat16)
    kwargs = (
        dict(raw_latent_shape=[1, 8, 10], patch_size=(1, 1, 1))
        if legacy
        else dict(video_spans=[{"start": 0, "latent_shape": [1, 8, 10]}])
    )
    impl(eqbsa).forward(q, q, q, meta(**kwargs))
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()


def test_missing_layout_rejected(eqbsa):
    q = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="explicit token layout"):
        impl(eqbsa).forward(q, q, q, None)
    eqbsa[1].assert_not_called()


def test_out_of_bounds_spans_rejected(eqbsa):
    q = torch.randn(1, 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="within the sequence"):
        impl(eqbsa).forward(
            q, q, q, meta(video_spans=[{"start": 1, "latent_shape": [1, 8, 16]}])
        )
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal": True},
        {"num_kv_heads": 1},
        {"dropout_p": 0.1},
        {"is_cross_attention": True},
        {"num_heads": 0},
    ],
)
def test_unsupported_semantics_rejected(eqbsa, kwargs):
    options = dict(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
    options.update(kwargs)
    with pytest.raises(ValueError):
        eqbsa[0].EQBSAAttentionImpl(**options)
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("head_size", [32, 64, 96, 256])
def test_unsupported_native_head_sizes_fail_before_dispatch(eqbsa, head_size):
    with pytest.raises(ValueError, match="head size 128"):
        eqbsa[0].EQBSAAttentionImpl(2, head_size, False, 0.125)
    eqbsa[1].assert_not_called()


def test_default_and_explicit_precision_select_only_mix():
    assert meta(video_spans=[]).precision == "mix"
    assert meta(video_spans=[], precision="mix").precision == "mix"
    assert EQBSAAttentionMetadata(current_timestep=0).precision == "mix"


@pytest.mark.parametrize("precision", ["bf16", "fp8", "mxfp4", "mxfp8", None])
def test_other_sparse_operator_modes_rejected(eqbsa, precision):
    with pytest.raises(ValueError, match="only supports precision='mix'"):
        meta(video_spans=[], precision=precision)
    with pytest.raises(ValueError, match="only supports precision='mix'"):
        EQBSAAttentionMetadata(current_timestep=0, precision=precision)
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("missing", ["precision", "video_spans", "both"])
def test_legacy_mindie_api_rejected_before_dispatch(eqbsa, monkeypatch, missing):
    def precision_only(q, k, v, *, precision="bf16", **kwargs):
        return eqbsa[1](q, k, v, **kwargs)

    def spans_only(q, k, v, *, video_spans=None, **kwargs):
        return eqbsa[1](q, k, v, **kwargs)

    def kwargs_only(q, k, v, **kwargs):
        return eqbsa[1](q, k, v, **kwargs)

    legacy_api = {
        "precision": spans_only,
        "video_spans": precision_only,
        "both": kwargs_only,
    }[missing]
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", legacy_api)
    with pytest.raises(ImportError, match="upgrade MindIE-SD"):
        impl(eqbsa)
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize(
    "path", ["sparse", "warmup", "no_video", "empty", "packed_empty"]
)
def test_mutated_precision_cannot_bypass_dispatch_validation(eqbsa, monkeypatch, path):
    metadata = meta(
        video_spans=(
            [] if path == "no_video" else [{"start": 0, "latent_shape": [1, 8, 16]}]
        )
    )
    if path == "warmup":
        metadata.current_timestep = 0
    metadata.precision = "bf16"
    adapter = impl(eqbsa)
    q = torch.empty(1, 0 if path == "empty" else 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="only supports precision='mix'"):
        if path == "packed_empty":
            context(monkeypatch, metadata)
            q = q[:, :0].reshape(0, 2, 128)
            adapter.forward_varlen(
                q, q, q, cu_seqlens=torch.tensor([0, 0]), max_seqlen=0
            )
        else:
            adapter.forward(q, q, q, metadata)
    eqbsa[1].assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "invalid",
    ["dtype", "mixed_dtype", "device", "head_size", "heads", "shape", "layout"],
)
def test_invalid_qkv_contract_fails_before_dispatch(eqbsa, packed, invalid):
    shape = (128, 2, 128) if packed else (1, 128, 2, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    k, v = q, q
    if invalid == "dtype":
        q = k = v = q.float()
    elif invalid == "mixed_dtype":
        k = k.float()
    elif invalid == "device":
        k = torch.empty(shape, dtype=q.dtype, device="meta")
    elif invalid == "head_size":
        q = k = v = q[..., :64]
    elif invalid == "heads":
        q = k = v = q[..., :0, :]
    elif invalid == "shape":
        k = k[..., :1, :]
    elif invalid == "layout":
        q = k = v = q.unsqueeze(0)
    with pytest.raises(ValueError):
        if packed:
            impl(eqbsa).forward_varlen(
                q, k, v, cu_seqlens=torch.tensor([0, 128]), max_seqlen=128
            )
        else:
            impl(eqbsa).forward(q, k, v, meta(video_spans=[]))
    eqbsa[1].assert_not_called()


def test_ulysses_partitioned_head_count_is_supported(eqbsa):
    adapter = eqbsa[0].EQBSAAttentionImpl(8, 128, False, 128**-0.5)
    q = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    adapter.forward(
        q, q, q, meta(video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}])
    )
    assert eqbsa[1].call_args.kwargs["head_num"] == 2
    assert eqbsa[1].call_args.kwargs["precision"] == "mix"


def test_layout_fallback_logs_its_reason(eqbsa, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(eqbsa[0].logger, "warning_once", warning)
    q = torch.zeros(1, 400, 2, 128, dtype=torch.bfloat16)
    spans = [
        {"start": 0, "latent_shape": [2, 10, 10]},
        {"start": 200, "latent_shape": [2, 10, 10]},
    ]
    impl(eqbsa).forward(q, q, q, meta(video_spans=spans))
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()
    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert "insufficient dense context" in message
    assert "operator=mindiesd.quant_attention" in message
    assert "precision=fp8" in message and "HIGH_PRECISION" in message


@pytest.mark.parametrize("dense", [False, True])
def test_batch_samples_have_independent_public_operator_calls(eqbsa, dense):
    # The external API stubs exercise layout/dispatch contracts on CPU. This is
    # not a numerical-equivalence claim about the native quantized operators.
    q = torch.zeros(2, 128, 2, 128, dtype=torch.bfloat16)
    v = torch.cat((torch.ones_like(q[:1]), torch.full_like(q[:1], 23.0)))
    adapter = impl(eqbsa)
    metadata = meta(
        current_timestep=0 if dense else 10,
        video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}],
    )

    output = adapter.forward(q, q, v, metadata)

    torch.testing.assert_close(output, v)
    operator = eqbsa[2] if dense else eqbsa[1]
    unused = eqbsa[1] if dense else eqbsa[2]
    unused.assert_not_called()
    assert operator.call_count == 2
    assert all(call.args[0].shape[0] == 1 for call in operator.call_args_list)
    assert all(
        tensor.is_contiguous()
        for call in operator.call_args_list
        for tensor in call.args
    )
    if dense:
        assert all(
            call.args[0].shape == (1, 2, 128, 128) for call in operator.call_args_list
        )
        assert all(call.kwargs == dense_options() for call in operator.call_args_list)
    else:
        assert all(
            call.kwargs["precision"] == "mix" for call in operator.call_args_list
        )
        assert all(
            call.kwargs["sparse_type"] == "rf_v3" for call in operator.call_args_list
        )


@pytest.mark.parametrize(
    "step, dense", [(0, True), (1, False), (2, False), (3, True), (4, True)]
)
def test_dense_tail_schedule_dispatches_by_step_index(eqbsa, step, dense):
    q = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    metadata = meta(
        current_timestep=step,
        skip_first_steps=1,
        dense_last_steps=2,
        num_steps=5,
        video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}],
    )
    impl(eqbsa).forward(q, q, q, metadata)
    if dense:
        eqbsa[1].assert_not_called()
        assert eqbsa[2].call_args.kwargs == dense_options()
    else:
        eqbsa[2].assert_not_called()
        assert eqbsa[1].call_args.kwargs["precision"] == "mix"


def test_oversized_dense_tail_clamps_to_all_steps(eqbsa):
    q = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    impl(eqbsa).forward(
        q,
        q,
        q,
        meta(
            current_timestep=1,
            skip_first_steps=0,
            dense_last_steps=20,
            num_steps=3,
            video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}],
        ),
    )
    eqbsa[1].assert_not_called()
    assert eqbsa[2].call_args.kwargs == dense_options()


@pytest.mark.parametrize(
    "changes",
    [
        {"dense_last_steps": 1},
        {"dense_last_steps": -1},
        {"dense_last_steps": True},
        {"dense_last_steps": 1.5},
        {"num_steps": 0},
        {"num_steps": True},
        {"current_timestep": 3, "num_steps": 3},
        {"current_timestep": True},
    ],
)
def test_invalid_tail_schedule_rejected(changes):
    with pytest.raises(ValueError):
        meta(video_spans=[], **changes)


def test_mutated_schedule_rejected_before_dense_dispatch(eqbsa):
    metadata = meta(video_spans=[])
    metadata.dense_last_steps = 1
    q = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires num_steps"):
        impl(eqbsa).forward(q, q, q, metadata)
    eqbsa[1].assert_not_called()
    eqbsa[2].assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
def test_runtime_heads_must_divide_configured_heads(eqbsa, packed):
    shape = (128, 3, 128) if packed else (1, 128, 3, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    adapter = impl(eqbsa)
    with pytest.raises(ValueError, match="dividing the configured"):
        if packed:
            adapter.forward_varlen(
                q, q, q, cu_seqlens=torch.tensor([0, 128]), max_seqlen=128
            )
        else:
            adapter.forward(q, q, q, meta(video_spans=[]))
    eqbsa[1].assert_not_called()
    eqbsa[2].assert_not_called()


@pytest.mark.parametrize(
    "host, maximum",
    [
        ((0, 2.5, 5), 3),
        ((0, True, 5), 5),
        ((0, 2, 5), True),
        ((0, 2, 5), 2),
        ((0, 5), 5),
    ],
)
def test_packed_host_bounds_and_maximum_are_strict(eqbsa, host, maximum):
    q = torch.zeros(5, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        impl(eqbsa).forward_varlen(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 2, 5]),
            cu_seqlens_host=host,
            max_seqlen=maximum,
        )
    eqbsa[1].assert_not_called()
    eqbsa[2].assert_not_called()


def test_h3_large_alignment_segment_remains_independent(eqbsa, monkeypatch):
    context(monkeypatch, meta(video_spans=[]))
    q = torch.zeros(64, 2, 128, dtype=torch.bfloat16)
    v = torch.cat((torch.ones_like(q[:5]), torch.full_like(q[5:], 31.0)))
    output = impl(eqbsa, packed_trailing_padding=True).forward_varlen(
        q, q, v, cu_seqlens=torch.tensor([0, 5, 64]), max_seqlen=5
    )
    torch.testing.assert_close(output, v)
    eqbsa[1].assert_not_called()
    assert [call.args[0].shape[2] for call in eqbsa[2].call_args_list] == [5, 59]


def test_dense_native_error_is_not_retried_as_sparse(eqbsa):
    eqbsa[2].side_effect = RuntimeError("native FIA error")
    q = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="native FIA"):
        impl(eqbsa).forward(q, q, q, meta(video_spans=[]))
    eqbsa[2].assert_called_once()
    eqbsa[1].assert_not_called()
