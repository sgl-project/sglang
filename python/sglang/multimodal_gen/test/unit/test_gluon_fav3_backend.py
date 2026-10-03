import sys
import types
from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import gluon_fav3
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _impl(**kwargs) -> gluon_fav3.GluonFAv3Impl:
    defaults = {
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_size": 128,
        "softmax_scale": 128**-0.5,
    }
    return gluon_fav3.GluonFAv3Impl(**(defaults | kwargs))


def test_backend_identity():
    assert gluon_fav3.GluonFAv3Backend.get_enum() is AttentionBackendEnum.GLUON_FAV3
    assert gluon_fav3.GluonFAv3Backend.get_impl_cls() is gluon_fav3.GluonFAv3Impl


def test_rejects_nondivisible_grouped_query_heads():
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        _impl(num_heads=8, num_kv_heads=3)


def test_forward_calls_gluon_kernel_for_supported_contract():
    impl = _impl()
    query = torch.empty(1, 2, 8, 128)
    key = torch.empty(1, 3, 8, 128)
    value = torch.empty_like(key)
    expected = torch.empty_like(query)
    kernel = mock.Mock(return_value=expected)

    with (
        mock.patch.object(impl, "_unsupported_reason", return_value=None),
        mock.patch.object(gluon_fav3, "_load_gluon_fav3", return_value=(kernel, None)),
    ):
        output = impl.forward(query, key, value)

    assert output is expected
    kernel.assert_called_once_with(
        query,
        key,
        value,
        softmax_scale=impl.softmax_scale,
    )


def test_unsupported_contract_uses_aiter_fallback():
    impl = _impl(causal=True)
    fallback = mock.Mock()
    expected = torch.empty(1)
    fallback.forward.return_value = expected
    impl._aiter_fallback = fallback
    query = torch.empty(1, 2, 8, 128)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    output = impl.forward(query, key, value)

    assert output is expected
    fallback.forward.assert_called_once_with(query, key, value, None)


def test_fallback_lazily_preserves_attention_configuration():
    impl = _impl(causal=True, dropout_p=0.25)
    fallback = mock.Mock()
    expected = torch.empty(1)
    fallback.forward.return_value = expected
    factory = mock.Mock(return_value=fallback)
    module_name = "sglang.multimodal_gen.runtime.layers.attention.backends.aiter"
    fake_module = types.ModuleType(module_name)
    fake_module.AITerImpl = factory
    query = torch.empty(1, 2, 8, 128)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    with mock.patch.dict(sys.modules, {module_name: fake_module}):
        output = impl.forward(query, key, value)

    assert output is expected
    factory.assert_called_once_with(
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        softmax_scale=impl.softmax_scale,
        causal=True,
        dropout_p=0.25,
    )


def test_runtime_head_sharding_remains_supported_for_mha():
    impl = _impl(num_heads=8, num_kv_heads=8)

    def tensor(shape):
        result = mock.Mock()
        result.ndim = 4
        result.dtype = torch.bfloat16
        result.is_cuda = True
        result.device = torch.device("cuda:0")
        result.shape = shape
        result.stride.return_value = 1
        return result

    query = tensor((1, 16, 4, 128))
    key = tensor((1, 16, 4, 128))
    value = tensor((1, 16, 4, 128))
    kernel = mock.Mock()

    with (
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=types.SimpleNamespace(gcnArchName="gfx1250"),
        ),
        mock.patch.object(gluon_fav3, "_load_gluon_fav3", return_value=(kernel, None)),
    ):
        assert impl._unsupported_reason(query, key, value) is None

    key.shape = (1, 16, 2, 128)
    value.shape = key.shape
    assert "runtime grouped-query attention" in impl._unsupported_reason(
        query, key, value
    )


def test_rocm_resolver_selects_gluon_only_for_exact_static_contract():
    backend_path = (
        "sglang.multimodal_gen.runtime.layers.attention.backends."
        "gluon_fav3.GluonFAv3Backend"
    )
    aiter_path = (
        "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend"
    )
    with mock.patch.object(
        RocmPlatform,
        "get_device_capability",
        return_value=DeviceCapability(12, 5),
    ):
        assert (
            RocmPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.GLUON_FAV3, 128, torch.bfloat16
            )
            == backend_path
        )
        assert (
            RocmPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.GLUON_FAV3, 128, torch.float16
            )
            == aiter_path
        )
        assert (
            RocmPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.GLUON_FAV3, 64, torch.bfloat16
            )
            == aiter_path
        )

    with mock.patch.object(
        RocmPlatform,
        "get_device_capability",
        return_value=DeviceCapability(9, 5),
    ):
        assert (
            RocmPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.GLUON_FAV3, 128, torch.bfloat16
            )
            == aiter_path
        )


def test_server_args_select_device_specific_rocm_default(monkeypatch):
    args = object.__new__(ServerArgs)
    platform_path = (
        "sglang.multimodal_gen.runtime.server_args.server_args.current_platform"
    )
    monkeypatch.delenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", raising=False)
    with (
        mock.patch(f"{platform_path}.is_rocm", return_value=True),
        mock.patch(
            f"{platform_path}.get_device_capability",
            return_value=DeviceCapability(12, 5),
        ),
    ):
        args._set_default_attention_backend()
    assert args.attention_backend == "aiter"

    monkeypatch.setenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", "1")
    with (
        mock.patch(f"{platform_path}.is_rocm", return_value=True),
        mock.patch(
            f"{platform_path}.get_device_capability",
            return_value=DeviceCapability(12, 5),
        ),
    ):
        args._set_default_attention_backend()
    assert args.attention_backend == "gluon_fav3"

    with (
        mock.patch(f"{platform_path}.is_rocm", return_value=True),
        mock.patch(
            f"{platform_path}.get_device_capability",
            return_value=DeviceCapability(9, 5),
        ),
    ):
        args._set_default_attention_backend()
    assert args.attention_backend == "aiter"


def test_server_args_accept_explicit_gluon_backend_name():
    assert ServerArgs._normalize_attention_backend_name("gluon_fav3") == "gluon_fav3"
