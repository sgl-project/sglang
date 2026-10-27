# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD adapter contracts, H3 wiring and opt-in native NPU validation."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, mock_open

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
    AscendFABackend,
    AscendFAImpl,
    AscendQuantAttention,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.npu import (
    NPUPlatformBase,
    load_mindiesd_quant_attention,
)


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
    from sglang.multimodal_gen.runtime import server_args

    config = {"precision": "fp8"}
    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config=config),
    )
    provider = Mock(side_effect=_reference)
    sdk = ModuleType("mindiesd")
    sdk.quant_attention = provider
    monkeypatch.setitem(sys.modules, "mindiesd", sdk)
    monkeypatch.setattr(AscendFAImpl, "_rot_matrices", {})
    monkeypatch.setattr(AscendQuantAttention, "_rot_matrices", {})

    def build(**kwargs):
        options = dict(num_heads=4, head_size=64, causal=False, softmax_scale=0.37)
        options.update(kwargs)
        return AscendFAImpl(**options)

    return config, provider, build


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.37])
def test_dispatch_preserves_inputs_and_rotates_once(
    adapter, precision, rotation, scale
):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=rotation)
    impl = build(softmax_scale=scale)
    query = torch.randn(1, 7, 4, 64)
    key, value = (torch.randn(1, 9, 2, 64) for _ in range(2))
    actual = impl.forward(query, key, value, None)
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
    torch.testing.assert_close(actual, _reference(query, key, value, **options))


def test_offline_rotation_is_respected_and_conflicts_rejected(adapter):
    config, provider, build = adapter
    offline = SimpleNamespace(use_offline_qk_rotation=True)
    impl = build(quant_config=offline)
    query = torch.randn(1, 7, 4, 64)
    impl.forward(query, query, query, None)
    assert provider.call_args.args[0] is query
    assert provider.call_args.kwargs["q_rot"] is None
    config["qk_rotation"] = True
    with pytest.raises(ValueError, match="offline"):
        build(quant_config=offline)


@pytest.mark.parametrize("head_size", [8, 64, 96, 128])
def test_rotation_matches_ascend_fa_and_preserves_rng(adapter, head_size):
    _, _, build = adapter
    state = torch.get_rng_state().clone()
    impl = build(head_size=head_size)._sd_executor
    assert torch.equal(state, torch.get_rng_state())
    AscendFAImpl._ensure_rot_matrix(head_size)
    rotation = impl._rot_matrices[head_size]
    assert torch.equal(rotation, AscendFAImpl._rot_matrices[head_size])
    torch.testing.assert_close(
        rotation @ rotation.T, torch.eye(head_size), atol=1e-6, rtol=1e-5
    )
    assert build(head_size=head_size)._sd_executor._rot_matrices[head_size] is rotation
    query = torch.zeros(1, 3, 4, head_size, dtype=torch.bfloat16)
    device_rotation = impl._get_rotation(query)
    assert impl._get_rotation(query) is device_rotation
    assert device_rotation.dtype == query.dtype
    assert impl._get_rotation(query.to(torch.float16)).dtype == torch.float16


def test_defaults_and_alternate_fp8_mode(adapter):
    config, provider, build = adapter
    impl = build(softmax_scale=None)._sd_executor
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
    with pytest.raises(ValueError, match=message):
        build(**kwargs)
    provider.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "int8"},
        {"fp8_fa_mode": "HIGH_PERFORMANCE"},
        {"fp8_fa_mode": "C8V16_TILING512"},
    ],
)
def test_sd_owns_precision_and_mode_constraints(adapter, options):
    config, provider, build = adapter
    config.update(options)
    impl = build(head_size=64)
    provider.assert_not_called()
    error = ValueError("SD precision/mode constraint")
    provider.side_effect = error
    query = torch.randn(1, 7, 4, 64)
    with pytest.raises(ValueError) as caught:
        impl.forward(query, query, query, None)
    assert caught.value is error
    provider.assert_called_once()
    for key, value in options.items():
        assert provider.call_args.kwargs[key] == value


@pytest.mark.parametrize("mode", ["high_precision", "c8v16_tiling512", None])
def test_fp8_mode_is_forwarded_without_normalization(adapter, mode):
    config, provider, build = adapter
    config["fp8_fa_mode"] = mode
    query = torch.randn(1, 7, 4, 128)
    build(head_size=128).forward(query, query, query, None)
    assert provider.call_args.kwargs["fp8_fa_mode"] == mode


@pytest.mark.parametrize("precision,calls", [("fp8", 3), ("mxfp8", 3), ("mxfp4", 1)])
def test_batch_execution_and_noncontiguous_inputs(adapter, precision, calls):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=False)
    query, key, value = (torch.randn(3, 4, 7, 64).transpose(1, 2) for _ in range(3))
    actual = build().forward(query, key, value, None)
    expected = _reference(query, key, value, scale=0.37, q_rot=None)
    torch.testing.assert_close(actual, expected)
    assert provider.call_count == calls


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize(
    "chunk,groups", [(0, [10]), (1, [1] * 10), (4, [4, 4, 2]), (8, [8, 2]), (16, [10])]
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("trailing_padding", [False, True])
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
    bounds = (0, 7, 8) if trailing_padding else (0, 0, 3, 8)
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_head_chunking_preserves_gqa_and_sdk_head_validation(adapter, precision):
    config, provider, build = adapter
    config.update(precision=precision, head_chunk_size=4, qk_rotation=False)
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_packed_segments_are_isolated_and_empty_segments_skipped(adapter, precision):
    config, provider, build = adapter
    config.update(precision=precision, qk_rotation=False)
    query = torch.zeros(7, 4, 64)
    value = torch.cat((torch.ones(3, 4, 64), torch.full((4, 4, 64), 23.0)))
    actual = build(packed_trailing_padding=True).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=torch.tensor([0, 0, 3, 7, 7], dtype=torch.int32),
        cu_seqlens_host=(0, 0, 3, 7, 7),
        max_seqlen=4,
    )
    torch.testing.assert_close(actual, value)
    assert provider.call_count == 2


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
    "bounds", [(1, 5), (0, 4), (0, 6, 5), (0,), (0, 5.0), (False, 5)]
)
def test_bad_packed_bounds(adapter, bounds):
    _, provider, build = adapter
    query = torch.randn(5, 4, 64)
    with pytest.raises(ValueError, match="monotonically"):
        build().forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.zeros(len(bounds), dtype=torch.int32),
            cu_seqlens_host=bounds,
            max_seqlen=5,
        )
    provider.assert_not_called()


@pytest.mark.parametrize("bounds", [torch.tensor([[0, 5]]), torch.tensor([0.0, 5.0])])
def test_bad_boundary_tensor(adapter, bounds):
    _, provider, build = adapter
    query = torch.randn(5, 4, 64)
    with pytest.raises(ValueError, match="1D int32"):
        build().forward_varlen(query, query, query, cu_seqlens=bounds, max_seqlen=5)
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("batch,length", [(0, 7), (1, 0), (2, 0)])
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


def test_native_errors_are_not_retried(adapter):
    _, provider, build = adapter
    provider.side_effect = RuntimeError("native failure")
    query = torch.randn(1, 7, 4, 64)
    with pytest.raises(RuntimeError, match="native failure"):
        build().forward(query, query, query, None)
    provider.assert_called_once()


def test_capabilities_and_lse(adapter):
    _, provider, build = adapter
    backend = AscendFABackend
    assert backend.get_enum() is AttentionBackendEnum.FA
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
    monkeypatch.setitem(
        sys.modules, "mindiesd", None if missing else ModuleType("mindiesd")
    )
    with pytest.raises(ImportError, match="MindIE-SD"):
        load_mindiesd_quant_attention()


@pytest.mark.parametrize("head_size", [8, 64, 96, 128, 256])
def test_npu_registration_defers_operator_support(adapter, monkeypatch, head_size):
    from sglang.multimodal_gen.runtime.platforms import npu

    soc_query = Mock(side_effect=AssertionError("registration must not query SoC"))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_soc_version=soc_query), raising=False
    )
    log = Mock()
    monkeypatch.setattr(npu.logger, "info", log)
    path = NPUPlatformBase.get_attn_backend_cls_str(
        AttentionBackendEnum.FA, head_size, torch.bfloat16
    )
    assert path.endswith("AscendFABackend")
    soc_query.assert_not_called()
    log.assert_called_once_with("Using Ascend Flash Attention backend.")


@pytest.mark.parametrize("head_size", [8, 96])
@pytest.mark.parametrize("packed", [False, True])
def test_head_dimension_reaches_sdk_and_native_errors_propagate(
    adapter, head_size, packed
):
    _, provider, build = adapter
    impl = build(head_size=head_size, softmax_scale=None)
    query = torch.randn(1, 7, 4, head_size)

    def forward():
        if packed:
            return impl.forward_varlen(
                query[0],
                query[0],
                query[0],
                cu_seqlens=torch.tensor([0, 7]),
                max_seqlen=7,
            )
        return impl.forward(query, query, query, None)

    # A mocked provider demonstrates adapter dispatch, not native D support.
    output = forward()
    assert output.shape == (query[0].shape if packed else query.shape)
    options = provider.call_args.kwargs
    assert options["scale"] == head_size**-0.5
    assert options["q_rot"].shape == (head_size, head_size)
    assert options["q_rot"] is options["k_rot"]
    provider.reset_mock()
    error = RuntimeError("native device/head-dimension constraint")
    provider.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        forward()
    assert caught.value is error
    provider.assert_called_once()


def test_only_fa_is_registered():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    assert "ASCEND_QUANT_FA" not in AttentionBackendEnum.__members__
    assert AscendFABackend.get_impl_cls() is AscendFAImpl
    with pytest.raises(ValueError, match="Invalid attention backend"):
        ServerArgs._normalize_attention_backend_name("ascend_quant_fa")


@pytest.mark.parametrize(
    "global_backend,components,roles,config,enabled,expected",
    [
        (None, {}, {}, {}, False, False),
        ("fa", {}, {}, {}, True, False),
        ("fa", {}, {}, {"precision": "fp8"}, False, True),
        ("fa3", {}, {}, {"precision": "mxfp4"}, False, True),
        (None, "transformer=fa", {}, {"precision": "fp8"}, False, True),
        (None, '{"transformer.self":"fa"}', {}, {"precision": "mxfp8"}, False, True),
        (None, {}, {"transformer": {"self": "fa"}}, {"precision": "fp8"}, False, True),
        ("torch_sdpa", {}, {}, {"precision": "fp8"}, False, False),
        ("torch_sdpa", {"transformer": "fa"}, {}, {"precision": "mxfp8"}, False, True),
    ],
)
def test_early_sdk_loading_covers_all_selection_forms(
    monkeypatch, global_backend, components, roles, config, enabled, expected
):
    from sglang.multimodal_gen.runtime.platforms import npu
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.attention_backend = global_backend
    args.attention_backend_config = config
    args.component_attention_backends = components
    args.component_attention_backend_roles = roles
    monkeypatch.setattr(npu.envs, "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", enabled)
    loader = Mock()
    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", loader)
    NPUPlatformBase().apply_server_args_defaults(args)
    assert loader.call_count == int(expected)
    assert args.component_attention_backends == components
    assert args.component_attention_backend_roles == roles


def test_early_loading_environment_is_inherited(monkeypatch):
    from sglang.multimodal_gen.runtime.platforms import npu
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.attention_backend = "fa"
    args.attention_backend_config = {"precision": "fp8"}
    args.component_attention_backends = {}
    args.component_attention_backend_roles = {}
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)

    def load():
        monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/test/mindiesd/ops")

    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", load)
    NPUPlatformBase().apply_server_args_defaults(args)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['ASCEND_CUSTOM_OPP_PATH'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == os.environ["ASCEND_CUSTOM_OPP_PATH"]


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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "invalid", ["dtype", "device", "gqa", "kv_shape", "head_dim", "empty_kv"]
)
def test_sdk_owns_common_tensor_validation(adapter, packed, invalid):
    _, provider, build = adapter
    query = torch.randn(1, 7, 4, 64)
    key, value = query.clone(), query.clone()
    if invalid == "dtype":
        key = key.half()
    elif invalid == "device":
        key = torch.empty_like(key, device="meta")
    elif invalid == "gqa":
        key, value = key[:, :, :3], value[:, :, :3]
    elif invalid == "kv_shape":
        value = value[:, :, :2]
    elif invalid == "head_dim":
        key, value = key[..., :32], value[..., :32]
    elif invalid == "empty_kv":
        if packed:
            # Packed token counts are adapter-owned; empty head axes are SDK-owned.
            key, value = key[:, :, :0], value[:, :, :0]
        else:
            key, value = key[:, :0], value[:, :0]
    error = ValueError("SDK input validation")
    provider.side_effect = error
    impl = build()
    with pytest.raises(ValueError) as caught:
        if packed:
            impl.forward_varlen(
                query[0],
                key[0],
                value[0],
                cu_seqlens=torch.tensor([0, 7]),
                max_seqlen=7,
            )
        else:
            impl.forward(query, key, value, None)
    assert caught.value is error
    provider.assert_called_once()


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
    return config, provider, build, quant, native_mx, native_float


@pytest.mark.parametrize("kind", ["MXFP8Config", "ModelSlimConfig"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("offline", [False, True])
def test_legacy_mxfp8_stays_native_with_sdk_installed(
    monkeypatch, legacy_native, kind, packed, offline
):
    from sglang.multimodal_gen.runtime.platforms import npu

    config, sd, build, quant, native_mx, native_float = legacy_native
    loader = Mock(side_effect=AssertionError("Legacy FA must not load MindIE-SD"))
    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", loader)
    qc = type(kind, (), {"use_offline_qk_rotation": offline})()
    impl = build(num_heads=6, quant_config=qc)
    assert impl._quant_scheme == "MXFP8" and impl._sd_executor is None
    assert impl.supports_pipelined_attention
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_precision_only_selects_sd_without_weight_or_environment_setup(
    monkeypatch, legacy_native, precision, enabled, packed
):
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa

    config, sd, build, quant, native_mx, native_float = legacy_native
    config.update(precision=precision, head_chunk_size=0)
    monkeypatch.setattr(
        ascend_fa.envs, "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", enabled
    )
    impl = build(num_heads=6, packed_trailing_padding=True)
    assert impl._sd_executor.precision == precision and impl._sd_executor.qk_rotation
    query = torch.randn(64, 6, 64)
    if packed:
        output = impl.forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.tensor([0, 63, 64]),
            cu_seqlens_host=(0, 63, 64),
            max_seqlen=63,
        )
        assert torch.count_nonzero(output[63:]) == 0
    else:
        impl.forward(query[None], query[None], query[None], None)
    sd.assert_called_once()
    assert sd.call_args.args[0].shape[2] == 6  # SD's explicit 0 overrides its default.
    assert sd.call_args.kwargs["precision"] == precision
    if precision == "fp8":
        assert sd.call_args.kwargs["fp8_fa_mode"] == "HIGH_PRECISION"
    for native in (quant, native_mx, native_float):
        native.assert_not_called()


@pytest.mark.parametrize("precision", ["none", "int8"])
def test_invalid_sd_precision_never_falls_back_to_native(legacy_native, precision):
    config, sd, build, quant, native_mx, native_float = legacy_native
    config.update(precision=precision)
    sd.side_effect = ValueError("Unsupported SD precision")
    impl = build(quant_config=type("MXFP8Config", (), {})())
    query = torch.randn(1, 64, 4, 64)
    with pytest.raises(ValueError, match="Unsupported SD precision"):
        impl.forward(query, query, query, None)
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


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize(
    "failure", [ImportError("missing SDK"), OSError("missing shared library")]
)
def test_sd_dependency_failure_never_falls_back(
    monkeypatch, legacy_native, precision, failure
):
    from sglang.multimodal_gen.runtime.platforms import npu

    config, sd, build, quant, native_mx, native_float = legacy_native
    config["precision"] = precision
    loader = Mock(side_effect=failure)
    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", loader)
    with pytest.raises(type(failure)) as caught:
        build(quant_config=type("MXFP8Config", (), {})())
    assert caught.value is failure
    loader.assert_called_once()
    for operator in (sd, quant, native_mx, native_float):
        operator.assert_not_called()


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_sd_execution_failure_never_retries_native(legacy_native, precision):
    config, sd, build, quant, native_mx, native_float = legacy_native
    config.update(precision=precision)
    impl = build()
    sd.side_effect = RuntimeError("operator execution failed")
    query = torch.randn(1, 64, 4, 64)
    with pytest.raises(RuntimeError, match="operator execution failed"):
        impl.forward(query, query, query, None)
    sd.assert_called_once()
    quant.assert_not_called()
    native_mx.assert_not_called()
    native_float.assert_not_called()


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_explicit_quantization_rejects_ring(adapter, precision):
    config, sd, build = adapter
    config["precision"] = precision
    impl = build()
    query = torch.randn(64, 4, 64)
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


@pytest.mark.parametrize("form", ["json", "pairs", "json_file", "yaml_file"])
def test_early_config_parsing_supports_all_cli_forms(monkeypatch, form):
    from sglang.multimodal_gen.runtime.platforms import npu
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    raw = '{"precision":"fp8","fp8_fa_mode":"C8V16_TILING512","head_chunk_size":4}'
    if form == "pairs":
        raw = "precision=fp8,fp8_fa_mode=C8V16_TILING512,head_chunk_size=4"
    elif form.endswith("file"):
        contents = (
            "precision: fp8\nfp8_fa_mode: C8V16_TILING512\nhead_chunk_size: 4\n"
            if form == "yaml_file"
            else raw
        )
        raw = "attention.yaml" if form == "yaml_file" else "attention.json"
        monkeypatch.setattr(os.path, "exists", lambda path: path == raw)
        monkeypatch.setattr("builtins.open", mock_open(read_data=contents))
    args = ServerArgs.__new__(ServerArgs)
    args.attention_backend = "fa"
    args.attention_backend_config = raw
    args.component_attention_backends = {}
    args.component_attention_backend_roles = {}
    loader = Mock()
    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", loader)
    NPUPlatformBase().apply_server_args_defaults(args)
    # Preloading must not bypass the normal string -> addict.Dict conversion.
    assert args.attention_backend_config == raw
    assert args._parse_attention_backend_config(raw) == {
        "precision": "fp8",
        "fp8_fa_mode": "C8V16_TILING512",
        "head_chunk_size": 4,
    }
    loader.assert_called_once()


def test_h3_sd_provider_disables_pipeline(monkeypatch):
    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _minimax_h3_pipelined_dense_attention,
    )

    monkeypatch.setattr(envs, "SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS", 2)
    attention = SimpleNamespace(
        _attention_backend_enum=AttentionBackendEnum.FA,
        _attention_impl=SimpleNamespace(supports_pipelined_attention=False),
    )
    query = torch.zeros(4, 4, 64)
    assert (
        _minimax_h3_pipelined_dense_attention(
            attention,
            query,
            query,
            query,
            cu_seqlens=torch.tensor([0, 4]),
            cu_seqlens_host=(0, 4),
            max_seqlen=4,
        )
        is None
    )


# Startup integration: exercise the real ServerArgs lifecycle without loading models.


@pytest.fixture
def make_server_args(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
        QwenImagePipelineConfig,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    for name, result in (
        ("is_cpu", False),
        ("is_mps", False),
        ("is_npu", False),
        ("is_cuda", True),
        ("is_rocm", False),
        ("get_device_total_memory", 80 * 1024**3),
        ("get_available_gpu_memory", 80),
        ("enable_dit_layerwise_offload_by_default", True),
        ("apply_server_args_defaults", None),
    ):
        monkeypatch.setattr(
            current_platform, name, lambda *args, result=result, **kwargs: result
        )

    def build(options, pipeline_config=None):
        config = pipeline_config or QwenImagePipelineConfig()
        with monkeypatch.context() as patch:
            patch.setattr(PipelineConfig, "from_kwargs", lambda *args, **kwargs: config)
            return ServerArgs.from_dict(options)

    return build


def test_mindiesd_loads_before_parameter_adjustment(monkeypatch, make_server_args):
    from sglang.multimodal_gen.runtime.platforms import current_platform, npu
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    adjust = ServerArgs._adjust_parameters
    load_sdk = Mock()
    monkeypatch.setattr(npu, "load_mindiesd_quant_attention", load_sdk)
    monkeypatch.setattr(
        current_platform,
        "apply_server_args_defaults",
        NPUPlatformBase().apply_server_args_defaults,
    )

    def check_then_adjust(args):
        # Parameter adjustment can query device memory and initialize NPU.
        load_sdk.assert_called_once()
        return adjust(args)

    monkeypatch.setattr(ServerArgs, "_adjust_parameters", check_then_adjust)
    make_server_args(
        {
            "model_path": "test/model",
            "attention_backend": "fa",
            "attention_backend_config": {"precision": "fp8"},
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"component_attention_backends": {"text_encoder": "fa"}},
        {"component_attention_backend_roles": {"text_encoder": {"self": "fa"}}},
    ],
)
@pytest.mark.parametrize("precision", [None, "fp8", "mxfp8", "mxfp4"])
def test_h3_text_encoder_default_preserves_overrides(
    monkeypatch, make_server_args, overrides, precision
):
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )

    monkeypatch.setattr(
        MiniMaxH3PipelineConfig, "validate_server_args", lambda *args, **kwargs: None
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    args = make_server_args(
        {
            "model_path": "/models/MiniMax-H3",
            "performance_mode": "manual",
            "attention_backend": "fa",
            "attention_backend_config": {}
            if precision is None
            else {"precision": precision},
            **overrides,
        },
        pipeline_config=MiniMaxH3PipelineConfig(),
    )
    assert args.attention_backend == "fa"
    if not overrides:
        if precision is not None:
            assert args.component_attention_backends["text_encoder"] == "torch_sdpa"
        else:
            assert "text_encoder" not in args.component_attention_backends
    elif "component_attention_backends" in overrides:
        assert args.component_attention_backends["text_encoder"] == "fa"
    else:
        assert args.component_attention_backend_roles["text_encoder"]["self"] == "fa"
        assert "text_encoder" not in args.component_attention_backends


# MiniMax-H3 integration: optional linear hooks and real backend dispatch.


@pytest.mark.parametrize("supported", [None, False, True])
def test_h3_mxfp8_input_hook_is_optional(supported):
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _accepts_mxfp8_input,
    )

    linear = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(linear)
    method = SimpleNamespace()
    if supported is not None:
        method.accepts_mxfp8_input = Mock(return_value=supported)
    linear.quant_method = method
    assert _accepts_mxfp8_input(linear) is (supported is True)
    if supported is not None:
        method.accepts_mxfp8_input.assert_called_once_with(linear)


@pytest.mark.parametrize(
    "enabled,tokens,packed,expected_mx",
    [
        (False, 64, False, False),
        (True, 65, False, False),
        (True, 64, False, True),
        (True, 128, True, True),
    ],
)
@pytest.mark.parametrize("runtime_heads", [2, 4])
def test_h3_quant_config_reaches_native_mxfp8_dispatch(
    monkeypatch, enabled, tokens, packed, expected_mx, runtime_heads
):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.layers.attention.backends import ascend_fa
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention

    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config={}),
    )

    # Use the real config and H3 installer. The smaller runtime head count
    # models ordinary Ulysses after its sequence-for-heads all-to-all.
    quant_config = MXFP8Config()
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 128
    attention.num_heads = 8
    attention.softmax_scale = 128**-0.5
    attention.prefix = "blocks.0.attn"
    attention._quant_config = quant_config
    monkeypatch.setattr(ascend_fa.current_platform, "is_npu", lambda: True)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_soc_version=lambda: 260), raising=False
    )
    monkeypatch.setattr(
        ascend_fa.envs, "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", enabled
    )
    monkeypatch.setattr(ascend_fa.envs, "SGLANG_DIFFUSION_MXFP8_FA_HEAD_CHUNK_SIZE", 0)
    # Older CPU torch builds lack E8M0; it is only a schema argument here.
    if not hasattr(torch, "float8_e8m0fnu"):
        monkeypatch.setattr(torch, "float8_e8m0fnu", torch.uint8, raising=False)
    quant = Mock(
        side_effect=lambda tensor, **kwargs: (
            tensor.to(kwargs["dst_type"]),
            torch.ones(1, dtype=torch.uint8),
        )
    )
    native_mx = Mock(
        side_effect=lambda q, k, v, **kwargs: (
            torch.zeros_like(q, dtype=kwargs["out_dtype"]),
        )
    )
    native_float = Mock(
        side_effect=lambda q, k, v, **kwargs: (torch.zeros_like(q), None)
    )
    monkeypatch.setattr(torch.ops.npu, "npu_dynamic_mx_quant", quant, raising=False)
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score_v2", native_mx, raising=False
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score", native_float, raising=False
    )
    attention._set_attention_backend(ascend_fa.AscendFABackend)
    impl = attention._attention_impl
    assert impl._quant_scheme == ("MXFP8" if enabled else None)
    q, k, v = (
        torch.randn(tokens, runtime_heads, 128, dtype=torch.bfloat16) for _ in range(3)
    )
    if packed:
        bounds = (0, 64, tokens)
        output = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
            cu_seqlens_host=bounds,
            max_seqlen=64,
        )
        expected_lengths = list(bounds[1:])
    else:
        output = impl.forward(q[None], k[None], v[None], None)[0]
        expected_lengths = [tokens]
    assert output.shape == q.shape
    if not expected_mx:
        quant.assert_not_called()
        native_mx.assert_not_called()
        native_float.assert_called_once()
        return
    native_float.assert_not_called()
    native_mx.assert_called_once()
    assert quant.call_count == 3
    rotation = impl._get_rotation(q.device, q.dtype)
    # The values at the quantization boundary contain exactly one Q/K
    # rotation; V is unchanged. Do not mock the MXFP8 helper itself.
    for call, expected, axis in zip(
        quant.call_args_list,
        (q @ rotation, k @ rotation, v),
        (-1, -1, 0),
    ):
        torch.testing.assert_close(call.args[0], expected)
        assert call.kwargs == {"dst_type": torch.float8_e4m3fn, "axis": axis}
    kwargs = native_mx.call_args.kwargs
    assert all(
        tensor.dtype == torch.float8_e4m3fn for tensor in native_mx.call_args.args
    )
    assert kwargs["input_layout"] == "TND"
    assert kwargs["num_query_heads"] == kwargs["num_key_value_heads"] == runtime_heads
    assert kwargs["actual_seq_qlen"] == kwargs["actual_seq_kvlen"] == expected_lengths
    assert kwargs["softmax_scale"] == attention.softmax_scale
    assert kwargs["query_quant_mode"] == kwargs["key_quant_mode"] == 6
    assert kwargs["value_quant_mode"] == 8
    assert kwargs["out_dtype"] == q.dtype


# A5 / D=64,128 is a validation matrix, not an adapter support whitelist.
# Enable these tests with SGLANG_TEST_ASCEND_QUANT_FA=1.


@pytest.fixture(scope="module")
def native():
    pytest.importorskip("torch_npu")
    from sglang.multimodal_gen.runtime.platforms.npu import (
        load_mindiesd_quant_attention,
    )

    # Import the SDK before initializing an NPU context.
    provider = load_mindiesd_quant_attention()
    if not torch.npu.is_available() or torch.npu.get_soc_version() < 260:
        pytest.skip("requires Ascend 950")
    torch.npu.set_device(0)
    return provider


@pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_ASCEND_QUANT_FA") != "1",
    reason="requires an explicitly enabled Ascend 950 / MindIE-SD environment",
)
class TestNativeAscendQuantFA:
    @pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
    @pytest.mark.parametrize("packed", [False, True])
    def test_native_head_chunking_matches_unchunked(
        self, native, monkeypatch, precision, packed
    ):
        from sglang.multimodal_gen.runtime import server_args

        config = dict(precision=precision, head_chunk_size=4)
        if precision == "fp8":
            config["fp8_fa_mode"] = "C8V16_TILING512"
        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(attention_backend_config=config),
        )
        chunked = AscendFAImpl(10, 128, False, 128**-0.5)
        config["head_chunk_size"] = 0
        unchunked = AscendFAImpl(10, 128, False, 128**-0.5)
        generator = torch.Generator().manual_seed(42)
        query, key, value = (
            torch.randn(2, 512, 10, 128, generator=generator, dtype=torch.bfloat16).to(
                "npu"
            )
            for _ in range(3)
        )
        if packed:
            query, key, value = (t.flatten(0, 1) for t in (query, key, value))
            options = dict(
                cu_seqlens=torch.tensor(
                    [0, 512, 1024], device="npu", dtype=torch.int32
                ),
                cu_seqlens_host=(0, 512, 1024),
                max_seqlen=512,
            )
            actual = chunked.forward_varlen(query, key, value, **options)
            expected = unchunked.forward_varlen(query, key, value, **options)
        else:
            actual = chunked.forward(query, key, value, None)
            expected = unchunked.forward(query, key, value, None)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("length", [32, 513])
    def test_native_mxfp8_batch_does_not_mix_value_groups(
        self, native, monkeypatch, length
    ):
        from sglang.multimodal_gen.runtime import server_args

        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(
                attention_backend_config={
                    "precision": "mxfp8",
                    "qk_rotation": False,
                }
            ),
        )
        impl = AscendFAImpl(4, 128, False, 128**-0.5)
        query = torch.zeros(2, length, 4, 128, device="npu", dtype=torch.bfloat16)
        # Powers of two are exactly representable in MXFP8. Uniform scores and
        # constant values give an independent analytical per-sequence reference.
        value = torch.stack((torch.ones_like(query[0]), torch.full_like(query[0], 16)))
        actual = impl.forward(query, query, value, None)
        torch.testing.assert_close(actual, value, atol=0.02, rtol=0.01)

    @pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
    @pytest.mark.parametrize("rotation", [False, True])
    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize("length", [512, 513])
    def test_native_dense_matches_public_sdk(
        self, native, monkeypatch, precision, rotation, head_size, length
    ):
        from sglang.multimodal_gen.runtime import server_args
        from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
            AscendFAImpl,
        )

        config = dict(precision=precision, qk_rotation=rotation)
        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(attention_backend_config=config),
        )
        impl = AscendFAImpl(4, head_size, False, head_size**-0.5)
        generator = torch.Generator().manual_seed(42)
        query, key, value = (
            torch.randn(
                2, length, 4, head_size, generator=generator, dtype=torch.bfloat16
            )
            .to("npu")
            .transpose(1, 2)
            .contiguous()
            .transpose(1, 2)
            for _ in range(3)
        )
        matrix = impl._sd_executor._get_rotation(query)
        options = dict(
            precision=precision,
            layout="BSND",
            scale=head_size**-0.5,
            q_rot=matrix,
            k_rot=matrix,
        )
        if precision == "fp8":
            options["fp8_fa_mode"] = "HIGH_PRECISION"
        if precision == "fp8" or (precision == "mxfp8" and length % 64):
            expected = torch.cat(
                [
                    native(
                        query[i : i + 1], key[i : i + 1], value[i : i + 1], **options
                    )
                    for i in range(2)
                ]
            )
        else:
            expected = native(query, key, value, **options)
        actual = impl.forward(query, key, value, None)
        assert actual.shape == query.shape and actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
    def test_native_packed_padding_and_gqa(self, native, monkeypatch, precision):
        from sglang.multimodal_gen.runtime import server_args
        from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
            AscendFAImpl,
        )

        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(attention_backend_config={"precision": precision}),
        )
        impl = AscendFAImpl(
            4, 128, False, 128**-0.5, num_kv_heads=2, packed_trailing_padding=True
        )
        query = torch.randn(576, 4, 128, device="npu", dtype=torch.bfloat16)
        key, value = (
            torch.randn(576, 2, 128, device="npu", dtype=torch.bfloat16)
            for _ in range(2)
        )
        actual = impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=torch.tensor([0, 512, 576], device="npu", dtype=torch.int32),
            cu_seqlens_host=(0, 512, 576),
            max_seqlen=512,
        )
        matrix = impl._sd_executor._get_rotation(query)
        options = {"fp8_fa_mode": "HIGH_PRECISION"} if precision == "fp8" else {}
        expected = native(
            query[:512].unsqueeze(0),
            key[:512].unsqueeze(0),
            value[:512].unsqueeze(0),
            precision=precision,
            layout="BSND",
            scale=128**-0.5,
            q_rot=matrix,
            k_rot=matrix,
            **options,
        )[0]
        torch.testing.assert_close(actual[:512], expected)
        assert torch.count_nonzero(actual[512:]) == 0

    @pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
    def test_native_fp16_asymmetric_gqa(self, native, monkeypatch, precision):
        from sglang.multimodal_gen.runtime import server_args
        from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
            AscendFAImpl,
        )

        config = {"precision": precision}
        if precision == "fp8":
            config["fp8_fa_mode"] = "C8V16_TILING512"
        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(attention_backend_config=config),
        )
        impl = AscendFAImpl(4, 128, False, 128**-0.5)
        query = torch.randn(1, 512, 4, 128, device="npu", dtype=torch.float16)
        key, value = (
            torch.randn(1, 513, 2, 128, device="npu", dtype=torch.float16)
            for _ in range(2)
        )
        matrix = impl._sd_executor._get_rotation(query)
        expected = native(
            query,
            key,
            value,
            layout="BSND",
            scale=128**-0.5,
            q_rot=matrix,
            k_rot=matrix,
            **config,
        )
        actual = impl.forward(query, key, value, None)
        torch.testing.assert_close(actual, expected)
        if precision == "mxfp4":
            assert actual.dtype == torch.bfloat16

    def test_native_c8v16_fp16(self, native, monkeypatch):
        from sglang.multimodal_gen.runtime import server_args
        from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
            AscendFAImpl,
        )

        monkeypatch.setattr(
            server_args,
            "get_global_server_args",
            lambda: SimpleNamespace(
                attention_backend_config={
                    "precision": "fp8",
                    "fp8_fa_mode": "C8V16_TILING512",
                }
            ),
        )
        impl = AscendFAImpl(4, 128, False, 128**-0.5)
        query, key, value = (
            torch.randn(1, 512, 4, 128, device="npu", dtype=torch.float16)
            for _ in range(3)
        )
        matrix = impl._sd_executor._get_rotation(query)
        expected = native(
            query,
            key,
            value,
            precision="fp8",
            layout="BSND",
            scale=128**-0.5,
            fp8_fa_mode="C8V16_TILING512",
            q_rot=matrix,
            k_rot=matrix,
        )
        torch.testing.assert_close(impl.forward(query, key, value, None), expected)
