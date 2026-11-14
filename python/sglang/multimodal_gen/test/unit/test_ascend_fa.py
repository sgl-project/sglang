# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Ascend FA dispatch and the MindIE-SD adapter."""

import sys
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
    AscendFABackend,
    AscendFAImpl,
    AscendFAMetadata,
    AscendFAMetadataBuilder,
    AscendQuantAttention,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase


def _reference(query, key, value, **options):
    if options["q_rot"] is not None:
        query = query @ options["q_rot"]
        key = key @ options["k_rot"]
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=options["scale"],
        enable_gqa=query.shape[2] != key.shape[2],
    ).transpose(1, 2)


@pytest.fixture
def adapter(monkeypatch):
    config = {"precision": "fp8"}
    provider = Mock(side_effect=_reference)
    sdk = ModuleType("mindiesd")
    sdk.quant_attention = provider
    monkeypatch.setitem(sys.modules, "mindiesd", sdk)
    monkeypatch.setattr(AscendFAImpl, "_rot_matrices", {})
    monkeypatch.setattr(AscendQuantAttention, "_rot_matrices", {})

    with ExitStack() as contexts:

        def build(**kwargs):
            metadata = AscendFAMetadataBuilder().build(attention_backend_config=config)
            contexts.enter_context(set_forward_context(0, metadata))
            options = dict(num_heads=4, head_size=64, causal=False, softmax_scale=0.37)
            options.update(kwargs)
            return AscendFAImpl(**options)

        yield config, provider, build


@pytest.mark.parametrize(
    "precision,rotation,scale",
    [("fp8", True, 0.0), ("mxfp8", False, 0.37), ("mxfp4", True, 0.37)],
)
def test_dispatch_preserves_inputs_and_rotates_once(
    adapter, precision, rotation, scale
):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=rotation)
    impl = build(softmax_scale=scale)
    query = torch.randn(1, 7, 4, 64)
    key, value = (torch.randn(1, 9, 2, 64) for _ in range(2))
    impl.forward(query, key, value, None)
    args, options = provider.call_args
    assert all(
        actual is expected for actual, expected in zip(args, (query, key, value))
    )
    assert options["precision"] == precision
    assert options["layout"] == "BSND"
    assert options["scale"] == scale
    assert options["pre_tokens"] == options["next_tokens"] == 2147483647
    assert options["q_rot"] is options["k_rot"]
    assert (options["q_rot"] is not None) == rotation
    assert ("fp8_fa_mode" in options) == (precision == "fp8")


def test_offline_rotation_is_respected_and_conflicts_rejected(adapter):
    config, provider, build = adapter
    offline = SimpleNamespace(use_offline_qk_rotation=True)
    impl = build(quant_config=offline)
    query = torch.randn(1, 7, 4, 64)
    impl.forward(query, query, query, None)
    assert provider.call_args.args[0] is query
    assert provider.call_args.kwargs["q_rot"] is None
    config["qk_rotation"] = True
    impl = build(quant_config=offline)
    with pytest.raises(ValueError, match="offline"):
        impl.forward(query, query, query, None)


def test_rotation_matches_ascend_fa_and_preserves_rng(adapter):
    head_size = 64
    _, _, build = adapter
    state = torch.get_rng_state().clone()
    impl = build(head_size=head_size)._get_sd_executor(
        get_forward_context().attn_metadata
    )
    assert torch.equal(state, torch.get_rng_state())
    AscendFAImpl._ensure_rot_matrix(head_size)
    rotation = impl._rot_matrices[head_size]
    assert torch.equal(rotation, AscendFAImpl._rot_matrices[head_size])
    torch.testing.assert_close(
        rotation @ rotation.T, torch.eye(head_size), atol=1e-6, rtol=1e-5
    )
    second = build(head_size=head_size)._get_sd_executor(
        get_forward_context().attn_metadata
    )
    assert second._rot_matrices[head_size] is rotation
    query = torch.zeros(1, 3, 4, head_size, dtype=torch.bfloat16)
    device_rotation = impl._get_rotation(query)
    assert impl._get_rotation(query) is device_rotation
    assert device_rotation.dtype == query.dtype
    assert impl._get_rotation(query.to(torch.float16)).dtype == torch.float16


def test_defaults_and_alternate_fp8_mode(adapter):
    config, provider, build = adapter
    impl = build(softmax_scale=None)._get_sd_executor(
        get_forward_context().attn_metadata
    )
    assert impl.precision == "fp8" and impl.qk_rotation
    assert impl.fp8_fa_mode == "HIGH_PRECISION"
    assert impl.head_chunk_size == 4
    assert impl.softmax_scale == 64**-0.5
    config["fp8_fa_mode"] = "C8V16_TILING512"
    query = torch.randn(1, 3, 4, 128)
    build(head_size=128).forward(query, query, query, None)
    assert provider.call_args.kwargs["fp8_fa_mode"] == "C8V16_TILING512"


@pytest.mark.parametrize(
    "config,kwargs,message",
    [
        ({"precision": "mxfp8", "fp8_fa_mode": None}, {}, "only supported"),
        ({"precision": "mxfp4", "fp8_fa_mode": "HIGH_PRECISION"}, {}, "only supported"),
        ({"qk_rotation": "false"}, {}, "boolean"),
        *[
            ({"head_chunk_size": size}, {}, "head_chunk_size")
            for size in (-1, True, 1.5, "4", None)
        ],
        ({}, {"causal": True}, "causal"),
        ({}, {"dropout_p": 0.1}, "dropout"),
    ],
)
def test_invalid_options_fail_before_provider(adapter, config, kwargs, message):
    options, provider, build = adapter
    options.update(config)
    query = torch.randn(1, 7, 4, 64)
    with pytest.raises(ValueError, match=message):
        build(**kwargs).forward(query, query, query, None)
    provider.assert_not_called()


@pytest.mark.parametrize(
    "precision,rotation,chunk,groups",
    [
        ("fp8", True, 4, [4, 4, 2]),
        ("mxfp8", False, 8, [8, 2]),
        ("mxfp4", True, 0, [10]),
        ("mxfp4", False, 16, [10]),
    ],
)
def test_head_chunks_preserve_dense_attention(
    adapter, precision, rotation, chunk, groups
):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=rotation, head_chunk_size=chunk)
    impl = build(num_heads=10)
    query = torch.randn(2, 10, 5, 64).transpose(1, 2)
    key, value = (torch.randn(2, 10, 7, 64).transpose(1, 2) for _ in range(2))
    snapshots = [t.clone() for t in (query, key, value)]
    actual = impl.forward(query, key, value, None)
    matrix = impl._sd_executor._get_rotation(query)
    expected = _reference(query, key, value, scale=0.37, q_rot=matrix, k_rot=matrix)
    torch.testing.assert_close(actual, expected)
    batch_calls = 2 if precision in ("fp8", "mxfp8") else 1
    assert [
        call.args[0].shape[2] for call in provider.call_args_list
    ] == groups * batch_calls
    for call in provider.call_args_list:
        assert "head_chunk_size" not in call.kwargs
        assert call.kwargs["precision"] == precision
        assert call.kwargs["q_rot"] is matrix
    for tensor, snapshot in zip((query, key, value), snapshots):
        torch.testing.assert_close(tensor, snapshot)


@pytest.mark.parametrize(
    "precision,trailing_padding", [("fp8", False), ("mxfp4", True)]
)
def test_head_chunks_preserve_packed_segments_and_dtype(
    adapter, precision, trailing_padding
):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=False, head_chunk_size=4)
    impl = build(num_heads=10, packed_trailing_padding=trailing_padding)
    provider.side_effect = lambda *args, **kw: _reference(*args, **kw).to(
        torch.bfloat16
    )
    query, key, value = (torch.randn(8, 10, 64) for _ in range(3))
    bounds = (0, 7, 8) if trailing_padding else (0, 0, 3, 8, 8)
    actual = impl.forward_varlen(
        query,
        key,
        value,
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
        cu_seqlens_host=bounds,
        max_seqlen=7 if trailing_padding else 5,
    )
    expected = torch.zeros_like(query, dtype=torch.bfloat16)
    segments = ((0, 7),) if trailing_padding else ((0, 3), (3, 8))
    for start, stop in segments:
        expected[start:stop] = _reference(
            query[None, start:stop],
            key[None, start:stop],
            value[None, start:stop],
            scale=0.37,
            q_rot=None,
            k_rot=None,
        )[0].to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.bfloat16
    assert [call.args[0].shape[2] for call in provider.call_args_list] == [
        4,
        4,
        2,
    ] * len(segments)


def test_head_chunking_preserves_gqa_and_sdk_head_validation(adapter):
    config, provider, build = adapter
    config.update(head_chunk_size=4, qk_rotation=False)
    impl = build(num_heads=10, num_kv_heads=2)
    query = torch.randn(1, 5, 10, 64)
    key, value = (torch.randn(1, 7, 2, 64) for _ in range(2))
    actual = impl.forward(query, key, value, None)
    torch.testing.assert_close(
        actual, _reference(query, key, value, scale=0.37, q_rot=None)
    )
    provider.assert_called_once()
    assert provider.call_args.args[0] is query
    provider.reset_mock()
    provider.side_effect = ValueError("SDK K/V head mismatch")
    key, value = torch.randn(1, 7, 10, 64), torch.randn(1, 7, 11, 64)
    with pytest.raises(ValueError, match="SDK K/V head mismatch"):
        impl.forward(query, key, value, None)
    provider.assert_called_once()
    assert provider.call_args.args[2] is value


@pytest.mark.parametrize("padded", [False, True])
def test_trailing_padding_requires_explicit_flag(adapter, padded):
    _, provider, build = adapter
    query = torch.zeros(7, 4, 64)
    value = torch.cat((torch.ones(3, 4, 64), torch.full((4, 4, 64), 23.0)))
    impl = build(packed_trailing_padding=padded)
    kwargs = dict(cu_seqlens=torch.tensor([0, 3, 7]), max_seqlen=3)
    if not padded:
        with pytest.raises(ValueError, match="max_seqlen"):
            impl.forward_varlen(query, query, value, **kwargs)
        provider.assert_not_called()
    else:
        output = impl.forward_varlen(query, query, value, **kwargs)
        torch.testing.assert_close(output[:3], value[:3])
        assert torch.count_nonzero(output[3:]) == 0
        provider.assert_called_once()


def test_host_bounds_avoid_device_read(adapter, monkeypatch):
    _, _, build = adapter
    query = torch.randn(5, 4, 64)
    bounds = torch.tensor([0, 5])
    monkeypatch.setattr(
        torch.Tensor, "tolist", Mock(side_effect=AssertionError("device read"))
    )
    build().forward_varlen(
        query, query, query, cu_seqlens=bounds, cu_seqlens_host=(0, 5), max_seqlen=5
    )


@pytest.mark.parametrize(
    "bounds,host_bounds,message",
    [
        *[
            (torch.zeros(len(bounds), dtype=torch.int32), bounds, "monotonically")
            for bounds in [(1, 5), (0, 4), (0, 6, 5), (0,), (0, 5.0), (False, 5)]
        ],
        (torch.tensor([[0, 5]]), None, "1D int32"),
        (torch.tensor([0.0, 5.0]), None, "1D int32"),
    ],
)
def test_bad_packed_bounds(adapter, bounds, host_bounds, message):
    _, provider, build = adapter
    query = torch.randn(5, 4, 64)
    with pytest.raises(ValueError, match=message):
        build().forward_varlen(
            query,
            query,
            query,
            cu_seqlens=bounds,
            cu_seqlens_host=host_bounds,
            max_seqlen=5,
        )
    provider.assert_not_called()


def test_packed_empty_and_all_padding_do_not_execute(adapter):
    _, provider, build = adapter
    impl = build(packed_trailing_padding=True)
    for total in (0, 7):
        query = torch.empty(total, 4, 64)
        out = impl.forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 0, total]), max_seqlen=0
        )
        assert out.shape == query.shape
        assert torch.count_nonzero(out) == 0
    provider.assert_not_called()


@pytest.mark.parametrize(
    "precision,batch,length", [("fp8", 0, 7), ("mxfp8", 1, 0), ("fp8", 2, 0)]
)
def test_dense_empty_inputs_reach_sd_validation(adapter, precision, batch, length):
    config, provider, build = adapter
    config["precision"] = precision
    error = ValueError("SD requires non-empty input axes")
    provider.side_effect = error
    query = torch.empty(batch, length, 4, 64)
    with pytest.raises(ValueError) as caught:
        build().forward(query, query, query, None)
    assert caught.value is error
    provider.assert_called_once()


def test_mxfp4_preserves_provider_dtype_and_unaligned_lengths(adapter):
    config, provider, build = adapter
    config.update(precision="mxfp4", qk_rotation=False)
    provider.side_effect = lambda q, k, v, **kw: torch.ones_like(
        q, dtype=torch.bfloat16
    )
    query = torch.randn(1, 513, 4, 64, dtype=torch.float16)
    impl = build()
    assert impl.forward(query, query, query, None).dtype == torch.bfloat16
    assert provider.call_args.args[0].shape[1] == 513
    packed = query[0]
    actual = impl.forward_varlen(
        packed, packed, packed, cu_seqlens=torch.tensor([0, 7, 513]), max_seqlen=506
    )
    assert actual.dtype == torch.bfloat16
    assert [call.args[0].shape[1] for call in provider.call_args_list] == [513, 7, 506]


def test_capabilities_and_lse(adapter):
    _, provider, build = adapter
    backend = AscendFABackend
    assert backend.get_enum() is AttentionBackendEnum.FA
    assert backend.get_impl_cls() is AscendFAImpl
    assert "ASCEND_QUANT_FA" not in AttentionBackendEnum.__members__
    assert not backend.get_enum().is_sparse
    assert (
        backend.unsupported_requirements(AttentionRequirements(packed_varlen=True))
        == ()
    )
    assert (
        backend.supports_ring_rotation()
    )  # The native floating-point path supports Ring.
    query = torch.randn(1, 7, 4, 64)
    with pytest.raises(NotImplementedError, match="LSE"):
        build().forward(query, query, query, None, return_softmax_lse=True)
    provider.assert_not_called()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_sdk_or_api(monkeypatch, missing):
    from sglang.multimodal_gen.runtime import server_args

    monkeypatch.setitem(
        sys.modules, "mindiesd", None if missing else ModuleType("mindiesd")
    )
    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config={"precision": "fp8"}),
    )
    with pytest.raises(ImportError, match="MindIE-SD"):
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.FA, 128, torch.bfloat16
        )


@pytest.mark.parametrize(
    "selected_backend,config,expected",
    [
        (AttentionBackendEnum.FA, None, False),
        (AttentionBackendEnum.FA, {}, False),
        (AttentionBackendEnum.FA, {"precision": "bf16"}, False),
        (AttentionBackendEnum.FA, {"precision": "fp8"}, True),
        (AttentionBackendEnum.FA, {"precision": "mxfp8"}, True),
        (AttentionBackendEnum.FA, {"precision": "mxfp4"}, True),
        (AttentionBackendEnum.TORCH_SDPA, {"precision": "fp8"}, False),
        (None, {"precision": "fp8"}, False),
    ],
)
def test_fa_selection_loads_sdk_only_for_quantization(
    monkeypatch, selected_backend, config, expected
):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    get_args = Mock(return_value=SimpleNamespace(attention_backend_config=config))
    if config is None:
        get_args.side_effect = ValueError("Global sgl_diffusion args is not set.")
    monkeypatch.setattr(server_args, "get_global_server_args", get_args)
    loader = Mock()
    monkeypatch.setattr(ascend_fa, "load_mindiesd_quant_attention", loader)
    backend = NPUPlatformBase.get_attn_backend_cls_str(
        selected_backend, 128, torch.bfloat16
    )
    assert loader.call_count == int(expected)
    assert backend.endswith(
        "AscendFABackend"
        if selected_backend == AttentionBackendEnum.FA
        else "SDPABackend"
    )


def test_invalid_adapter_layout_or_head_size(adapter):
    _, provider, build = adapter
    impl = build()
    query = torch.zeros(1, 7, 4, 64)
    cases = [
        (query[0], query, query),
        (query[..., :32], query[..., :32], query[..., :32]),
    ]
    for q, k, v in cases:
        with pytest.raises(ValueError):
            impl.forward(q, k, v, None)
    provider.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("oversized", ["key", "value"])
def test_slicing_cannot_hide_extra_batches_or_tokens(adapter, packed, oversized):
    _, provider, build = adapter
    shape = (7, 4, 64) if packed else (2, 7, 4, 64)
    tensors = {name: torch.randn(shape) for name in ("query", "key", "value")}
    tensors[oversized] = torch.cat((tensors[oversized], tensors[oversized][:1]))
    impl = build()
    with pytest.raises(ValueError, match="batch sizes or packed token counts"):
        if packed:
            impl.forward_varlen(
                **tensors, cu_seqlens=torch.tensor([0, 3, 7]), max_seqlen=4
            )
        else:
            impl.forward(**tensors, attn_metadata=None)
    provider.assert_not_called()


@pytest.fixture
def legacy_native(monkeypatch, adapter):
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    config, provider, build = adapter
    config.clear()
    monkeypatch.setattr(ascend_fa.current_platform, "is_npu", lambda: True)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_soc_version=lambda: 260), raising=False
    )
    monkeypatch.setattr(ascend_fa.envs, "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", True)
    monkeypatch.setattr(ascend_fa.envs, "SGLANG_DIFFUSION_MXFP8_FA_HEAD_CHUNK_SIZE", 4)
    monkeypatch.setattr(torch, "float8_e8m0fnu", torch.uint8, raising=False)
    quant = Mock(side_effect=lambda tensor, **kw: (tensor, torch.ones(1)))
    native_mx = Mock(side_effect=lambda q, *a, **kw: (torch.zeros_like(q), None))
    native_float = Mock(side_effect=lambda q, *a, **kw: (torch.zeros_like(q), None))
    monkeypatch.setattr(torch.ops.npu, "npu_dynamic_mx_quant", quant, raising=False)
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score_v2", native_mx, raising=False
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score", native_float, raising=False
    )
    # The packed native entry point checks NPU devices before invoking the op.
    monkeypatch.setattr(
        ascend_fa,
        "fused_infer_attention_varlen",
        lambda q, k, v, *args, **kwargs: native_float(q, k, v, **kwargs)[0],
    )
    return config, provider, build, quant, native_mx, native_float


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scope", ["absent", "unrelated", "empty", "bf16", "cross"])
def test_metadata_scope_preserves_native_dispatch(
    monkeypatch, legacy_native, packed, scope
):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    config, sd, build, quant, native_mx, native_float = legacy_native
    config["precision"] = "fp8"
    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        Mock(side_effect=AssertionError("FA must use forward metadata")),
    )
    loader = Mock(side_effect=AssertionError("Native FA must not load MindIE-SD"))
    monkeypatch.setattr(ascend_fa, "load_mindiesd_quant_attention", loader)
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    impl = build(
        quant_config=type("MXFP8Config", (), {})(),
        is_cross_attention=scope == "cross",
    )
    assert impl._sd_executor is None
    metadata = {
        "absent": None,
        "unrelated": SimpleNamespace(precision="mxfp4"),
        "empty": AscendFAMetadata(),
        "bf16": AscendFAMetadata(precision="bf16"),
        "cross": AscendFAMetadata(precision="fp8"),
    }[scope]
    query = torch.randn(64, 4, 64)
    with set_forward_context(0, metadata):
        if packed:
            actual = impl.forward_varlen(
                query, query, query, cu_seqlens=torch.tensor([0, 64]), max_seqlen=64
            )
        else:
            actual = impl.forward(query[None], query[None], query[None], None)
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    expected_mx = scope in ("absent", "unrelated", "empty")
    assert native_mx.call_count == int(expected_mx)
    assert quant.call_count == 3 * int(expected_mx)
    assert native_float.call_count == int(not expected_mx)
    sd.assert_not_called()
    loader.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
def test_metadata_changes_do_not_leak_between_calls(legacy_native, packed):
    config, sd, build, _, native_mx, native_float = legacy_native
    config["precision"] = "mxfp8"
    impl = build()
    query = torch.randn(64, 4, 64)
    configs = [
        {"precision": "fp8", "qk_rotation": False},
        {"precision": "mxfp4", "head_chunk_size": 2, "qk_rotation": True},
        {"precision": "bf16"},
        {"precision": "fp8", "qk_rotation": False},
    ]
    for config in configs:
        metadata = AscendFAMetadataBuilder().build(attention_backend_config=config)
        sd.reset_mock()
        if packed:
            actual = impl.forward_varlen(
                query,
                query,
                query,
                cu_seqlens=torch.tensor([0, 64]),
                max_seqlen=64,
                attn_metadata=metadata,
            )
        else:
            actual = impl.forward(query[None], query[None], query[None], metadata)[0]
        if config["precision"] == "bf16":
            sd.assert_not_called()
            torch.testing.assert_close(actual, torch.zeros_like(query))
        else:
            options = sd.call_args.kwargs
            assert options["precision"] == config["precision"]
            assert (options["q_rot"] is not None) == config["qk_rotation"]
            assert sd.call_count == (2 if config["precision"] == "mxfp4" else 1)
            torch.testing.assert_close(
                actual, _reference(query[None], query[None], query[None], **options)[0]
            )
    native_float.assert_called_once()
    native_mx.assert_not_called()


@pytest.mark.parametrize(
    "kind,packed,offline",
    [("MXFP8Config", False, False), ("ModelSlimConfig", True, True)],
)
def test_legacy_mxfp8_stays_native_with_sdk_installed(
    monkeypatch, legacy_native, kind, packed, offline
):
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    config, sd, build, quant, native_mx, native_float = legacy_native
    loader = Mock(side_effect=AssertionError("Legacy FA must not load MindIE-SD"))
    monkeypatch.setattr(ascend_fa, "load_mindiesd_quant_attention", loader)
    qc = type(kind, (), {"use_offline_qk_rotation": offline})()
    # Ulysses can reduce the head count passed to the kernel.
    impl = build(num_heads=12, quant_config=qc)
    assert impl._quant_scheme == "MXFP8" and impl._sd_executor is None
    query = torch.randn(128, 6, 64)
    original = query.clone()
    if packed:
        output = impl.forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.tensor([0, 64, 128]),
            cu_seqlens_host=(0, 64, 128),
            max_seqlen=64,
        )
    else:
        output = impl.forward(query[None], query[None], query[None], None)
    assert output.shape == (query.shape if packed else query[None].shape)
    assert [call.args[0].shape[1] for call in native_mx.call_args_list] == [4, 2]
    lengths = [64, 128] if packed else [128]
    for call in native_mx.call_args_list:
        options = call.kwargs
        assert (
            options["num_query_heads"]
            == options["num_key_value_heads"]
            == call.args[0].shape[1]
        )
        assert options["actual_seq_qlen"] == options["actual_seq_kvlen"] == lengths
        assert options["softmax_scale"] == impl.softmax_scale
    expected = (
        query if offline else query @ impl._get_rotation(query.device, query.dtype)
    )
    torch.testing.assert_close(quant.call_args_list[0].args[0], expected[:, :4])
    torch.testing.assert_close(quant.call_args_list[3].args[0], expected[:, 4:])
    torch.testing.assert_close(query, original)
    assert quant.call_count == 6
    loader.assert_not_called()
    sd.assert_not_called()
    native_float.assert_not_called()


@pytest.mark.parametrize(
    "precision,error",
    [
        ("none", ValueError("Unsupported SD precision")),
        ("fp8", RuntimeError("operator execution failed")),
    ],
)
def test_sd_execution_failure_never_retries_native(legacy_native, precision, error):
    config, sd, build, quant, native_mx, native_float = legacy_native
    config.update(precision=precision)
    sd.side_effect = error
    impl = build(quant_config=type("MXFP8Config", (), {})())
    query = torch.randn(1, 64, 4, 64)
    with pytest.raises(type(error)) as caught:
        impl.forward(query, query, query, None)
    assert caught.value is error
    sd.assert_called_once()
    assert sd.call_args.kwargs["precision"] == precision
    for native in (quant, native_mx, native_float):
        native.assert_not_called()


@pytest.mark.parametrize(
    "case", ["disabled", "unquantized", "causal", "cross", "unaligned"]
)
def test_legacy_nonquantized_calls_remain_native_float(
    monkeypatch, legacy_native, case
):
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    _, sd, build, quant, native_mx, native_float = legacy_native
    qc = None if case == "unquantized" else type("MXFP8Config", (), {})()
    if case == "disabled":
        monkeypatch.setattr(
            ascend_fa.envs, "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", False
        )
    impl = build(
        quant_config=qc, causal=case == "causal", is_cross_attention=case == "cross"
    )
    query = torch.randn(1, 65 if case == "unaligned" else 64, 4, 64)
    impl.forward(query, query, query, None)
    sd.assert_not_called()
    quant.assert_not_called()
    native_mx.assert_not_called()
    native_float.assert_called_once()


@pytest.mark.parametrize(
    "failure", [ImportError("missing SDK"), OSError("missing shared library")]
)
def test_sd_dependency_failure_never_falls_back(monkeypatch, legacy_native, failure):
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    config, sd, build, quant, native_mx, native_float = legacy_native
    config["precision"] = "fp8"
    loader = Mock(side_effect=failure)
    monkeypatch.setattr(ascend_fa, "load_mindiesd_quant_attention", loader)
    impl = build(quant_config=type("MXFP8Config", (), {})())
    loader.assert_not_called()
    query = torch.randn(1, 64, 4, 64)
    with pytest.raises(type(failure)) as caught:
        impl.forward(query, query, query, None)
    assert caught.value is failure
    loader.assert_called_once()
    for operator in (sd, quant, native_mx, native_float):
        operator.assert_not_called()


def test_explicit_quantization_rejects_ring(adapter):
    config, sd, build = adapter
    impl = build()
    query = torch.randn(64, 4, 64)
    with pytest.raises(ValueError, match="Ring/LSE"):
        AscendFAMetadataBuilder().build(attention_backend_config=config, ring_degree=2)
    with pytest.raises(NotImplementedError, match="Ring/LSE"):
        impl.forward_ring_kv_chunk(query, query, query)
    sd.assert_not_called()


def test_unaligned_mxfp8_batch_uses_independent_sequences(adapter):
    config, sd, build = adapter
    config.update(precision="mxfp8", qk_rotation=False)
    query = torch.zeros(2, 32, 4, 64)
    value = torch.stack((torch.ones_like(query[0]), torch.full_like(query[0], 16)))
    actual = build().forward(query, query, value, None)
    torch.testing.assert_close(actual, value)
    assert sd.call_count == 2
    assert all(call.args[0].shape[0] == 1 for call in sd.call_args_list)


@pytest.mark.parametrize("precision,ring_degree", [("fp8", 1), ("bf16", 2), (None, 2)])
def test_denoising_builds_scoped_fa_metadata(precision, ring_degree):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    config = {"head_chunk_size": 2, "qk_rotation": False}
    if precision is not None:
        config["precision"] = precision
    stage = SimpleNamespace(attn_backend=AscendFABackend)
    metadata = DenoisingStage._build_attn_metadata(
        stage,
        0,
        SimpleNamespace(raw_latent_shape=(1, 16, 2, 8, 8)),
        SimpleNamespace(attention_backend_config=config, ring_degree=ring_degree),
    )
    assert metadata == AscendFAMetadata(
        precision=precision,
        head_chunk_size=2,
        qk_rotation=False,
    )
    config["precision"] = "mxfp4"
    assert metadata.precision == precision
