import sys
import types
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.gluon import flash_attention as gluon_dispatch
from sglang.kernels.ops.attention.gluon.fa3 import gfx1250 as gfx1250_impl
from sglang.multimodal_gen.runtime.layers.attention import selector
from sglang.multimodal_gen.runtime.layers.attention.backends import gluon_fa
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _impl(**kwargs) -> gluon_fa.GluonFAImpl:
    defaults = {
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_size": 128,
        "softmax_scale": 128**-0.5,
    }
    return gluon_fa.GluonFAImpl(**(defaults | kwargs))


def test_backend_identity():
    assert gluon_fa.GluonFABackend.get_enum() is AttentionBackendEnum.GLUON_FA
    assert gluon_fa.GluonFABackend.get_impl_cls() is gluon_fa.GluonFAImpl


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
        mock.patch.object(
            gluon_fa, "get_flash_attention_impl", return_value=(kernel, None)
        ),
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_runtime_head_sharding_remains_supported_for_mha(dtype, head_size):
    impl = _impl(
        num_heads=8, num_kv_heads=8, head_size=head_size, softmax_scale=head_size**-0.5
    )

    def tensor(shape):
        result = mock.Mock()
        result.ndim = 4
        result.dtype = dtype
        result.is_cuda = True
        result.device = torch.device("cuda:0")
        result.shape = shape
        result.stride.return_value = 1
        return result

    query = tensor((1, 16, 4, head_size))
    key = tensor((1, 16, 4, head_size))
    value = tensor((1, 16, 4, head_size))
    kernel = mock.Mock()

    with (
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=types.SimpleNamespace(gcnArchName="gfx1250"),
        ),
        mock.patch.object(
            gluon_fa, "get_flash_attention_impl", return_value=(kernel, None)
        ),
    ):
        assert impl._unsupported_reason(query, key, value) is None

    key.shape = (1, 16, 2, head_size)
    value.shape = key.shape
    assert "runtime grouped-query attention" in impl._unsupported_reason(
        query, key, value
    )


@pytest.mark.parametrize(
    ("capability", "head_size", "dtype"),
    [
        (DeviceCapability(12, 5), 128, torch.bfloat16),
        (DeviceCapability(12, 5), 128, torch.float16),
        (DeviceCapability(12, 5), 64, torch.bfloat16),
        (DeviceCapability(9, 5), 128, torch.bfloat16),
        (None, 128, torch.bfloat16),
    ],
)
def test_component_override_preserves_gluon_identity(capability, head_size, dtype):
    selector._cached_get_attn_backend.cache_clear()
    try:
        with (
            mock.patch.object(
                RocmPlatform, "get_device_capability", return_value=capability
            ),
            mock.patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform",
                RocmPlatform,
            ),
            mock.patch.object(
                selector, "get_global_forced_attn_backend", return_value=None
            ),
            selector.component_attn_backend_context_manager(
                AttentionBackendEnum.GLUON_FA,
                component_name="transformer",
                require_backend_selection=True,
            ),
        ):
            backend = selector.get_attn_backend(head_size, dtype)

        assert backend is gluon_fa.GluonFABackend
    finally:
        selector._cached_get_attn_backend.cache_clear()


def test_server_args_keep_aiter_as_rocm_default():
    args = object.__new__(ServerArgs)
    platform_path = (
        "sglang.multimodal_gen.runtime.server_args.server_args.current_platform"
    )
    with (
        mock.patch(f"{platform_path}.is_rocm", return_value=True),
        mock.patch(
            f"{platform_path}.get_device_capability",
            return_value=DeviceCapability(12, 5),
        ),
    ):
        args._set_default_attention_backend()
    assert args.attention_backend == "aiter"

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
    assert ServerArgs._normalize_attention_backend_name("gluon_fa") == "gluon_fa"


def test_aiter_fallback_passes_nondefault_scale_to_dense_kernel():
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter

    impl = _impl(causal=True, dropout_p=0.25, softmax_scale=0.37)
    query = torch.empty(1, 2, 8, 128, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    expected = torch.empty_like(query)

    with (
        mock.patch.object(aiter, "_use_fp8_attn", False),
        mock.patch.object(
            aiter.aiter, "flash_attn_func", return_value=expected
        ) as flash_attn,
    ):
        output = impl.forward(query, key, value)

    assert output is expected
    flash_attn.assert_called_once_with(
        query,
        key,
        value,
        dropout_p=0.25,
        softmax_scale=0.37,
        causal=True,
        return_attn_probs=False,
        return_lse=False,
        how_v3_bf16_cvt=aiter.AITER_BF16_CVT_MODE,
    )


@pytest.fixture
def gluon_modules():
    kernel_name = "sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"
    modules = {
        "triton.experimental.gluon": mock.Mock(),
        "triton.experimental.gluon.language": mock.Mock(),
        kernel_name: types.SimpleNamespace(flash_attention=mock.Mock()),
    }
    original_import = gfx1250_impl.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name not in modules:
            return original_import(name, *args, **kwargs)
        result = modules[name]
        if isinstance(result, Exception):
            raise result
        return result

    gluon_dispatch._load_implementation.cache_clear()
    try:
        with (
            mock.patch.object(
                gfx1250_impl.importlib, "import_module", side_effect=import_module
            ) as loader,
            mock.patch.object(gluon_fa, "_logged_fallback_reasons", set()),
            mock.patch.object(torch.version, "hip", "test"),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=types.SimpleNamespace(gcnArchName="gfx1250"),
            ),
        ):
            yield modules, loader
    finally:
        gluon_dispatch._load_implementation.cache_clear()


@pytest.mark.parametrize(
    ("module_name", "api"),
    [
        ("triton.experimental.gluon", "aggregate"),
        ("triton.experimental.gluon.language", "amd.cdna5"),
        ("triton.experimental.gluon.language", "amd.cdna5.wmma"),
        ("triton.experimental.gluon.language", "amd.cdna5.buffer_load"),
        ("triton.experimental.gluon.language", "amd.cdna5.buffer_store"),
        ("triton.experimental.gluon.language", "amd.cdna5.tdm.async_load"),
        ("triton.experimental.gluon.language", "amd.cdna5.tdm.async_load_fused"),
        (
            "triton.experimental.gluon.language",
            "amd.cdna5.tdm.update_tensor_descriptor",
        ),
        ("triton.experimental.gluon.language", "amd.cdna5.tdm.prefetch"),
    ],
)
def test_missing_gluon_api_falls_back_before_kernel_import(
    gluon_modules, module_name, api, caplog
):
    modules, loader = gluon_modules
    missing = modules[module_name]
    path = api.split(".")
    for attribute in path[:-1]:
        missing = getattr(missing, attribute)
    delattr(missing, path[-1])

    impl = _impl()
    expected = torch.empty(1)
    impl._aiter_fallback = mock.Mock()
    impl._aiter_fallback.forward.return_value = expected
    query = mock.Mock(
        ndim=4,
        dtype=torch.bfloat16,
        is_cuda=True,
        device=torch.device("cuda:0"),
        shape=(1, 16, 8, 128),
    )
    query.stride.return_value = 1
    with mock.patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=types.SimpleNamespace(gcnArchName="gfx1250"),
    ):
        output = impl.forward(query, query, query)

    assert output is expected
    impl._aiter_fallback.forward.assert_called_once_with(query, query, query, None)
    assert f"{module_name}.{api}" in caplog.text
    kernel_name = "sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"
    assert kernel_name not in [call.args[0] for call in loader.call_args_list]


def test_compatible_gluon_apis_load_and_cache_kernel(gluon_modules):
    modules, loader = gluon_modules
    expected = modules[
        "sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"
    ].flash_attention

    assert gluon_dispatch.get_flash_attention_impl(torch.device("cuda:0")) == (
        expected,
        None,
    )
    assert gluon_dispatch.get_flash_attention_impl(torch.device("cuda:0")) == (
        expected,
        None,
    )
    assert loader.call_count == 3


def test_missing_gluon_dependency_is_reported(gluon_modules):
    modules, _ = gluon_modules
    modules["triton.experimental.gluon"] = ImportError("Gluon is unavailable")

    assert gluon_dispatch.get_flash_attention_impl(torch.device("cuda:0")) == (
        None,
        "Gluon is unavailable",
    )


@pytest.mark.parametrize("error", [AttributeError, RuntimeError])
def test_unexpected_kernel_import_errors_propagate(gluon_modules, error):
    modules, _ = gluon_modules
    modules["sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"] = error(
        "kernel bug"
    )

    with pytest.raises(error, match="kernel bug"):
        gluon_dispatch.get_flash_attention_impl(torch.device("cuda:0"))


@pytest.mark.parametrize("error", [AttributeError, RuntimeError])
def test_kernel_execution_errors_propagate(error):
    impl = _impl()
    impl._aiter_fallback = mock.Mock()
    query = torch.empty(1, 2, 8, 128)
    kernel = mock.Mock(side_effect=error("kernel bug"))

    with (
        mock.patch.object(impl, "_unsupported_reason", return_value=None),
        mock.patch.object(
            gluon_fa, "get_flash_attention_impl", return_value=(kernel, None)
        ),
        pytest.raises(error, match="kernel bug"),
    ):
        impl.forward(query, query, query)

    impl._aiter_fallback.forward.assert_not_called()


@pytest.mark.parametrize(
    "head_size,dtype,reason",
    [(32, torch.bfloat16, "head_size=32"), (128, torch.float32, "dtype=torch.float32")],
)
def test_extended_contract_keeps_unsupported_fallback(head_size, dtype, reason):
    impl = _impl(head_size=head_size)
    query = torch.empty(1, 2, 8, head_size, dtype=dtype)
    assert impl._unsupported_reason(query, query, query) == reason


@pytest.mark.parametrize("architecture", ["gfx950", "gfx942", "gfx12501", ""])
def test_unsupported_architecture_does_not_import_gluon(gluon_modules, architecture):
    _, loader = gluon_modules
    with mock.patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=types.SimpleNamespace(gcnArchName=architecture),
    ):
        implementation, reason = gluon_dispatch.get_flash_attention_impl(
            torch.device("cuda:0")
        )
    assert implementation is None
    assert reason == f"GPU architecture {architecture or 'unknown'}"
    loader.assert_not_called()


def test_dispatch_uses_requested_device_and_normalizes_architecture(gluon_modules):
    modules, loader = gluon_modules
    expected = modules[
        "sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"
    ].flash_attention

    def properties(device):
        architecture = "gfx1250:sramecc+:xnack-" if device.index == 1 else "gfx950"
        return types.SimpleNamespace(gcnArchName=architecture)

    with mock.patch.object(torch.cuda, "get_device_properties", side_effect=properties):
        for index in (0, 1, 0, 1):
            implementation, reason = gluon_dispatch.get_flash_attention_impl(
                torch.device(f"cuda:{index}")
            )
            if index == 1:
                assert implementation is expected and reason is None
            else:
                assert implementation is None and "gfx950" in reason
    assert loader.call_count == 3


@pytest.mark.parametrize("device,hip", [("cpu", "test"), ("cuda:0", None)])
def test_non_rocm_devices_do_not_query_gpu_or_import_gluon(gluon_modules, device, hip):
    _, loader = gluon_modules
    with (
        mock.patch.object(torch.version, "hip", hip),
        mock.patch.object(torch.cuda, "get_device_properties") as properties,
    ):
        implementation, reason = gluon_dispatch.get_flash_attention_impl(
            torch.device(device)
        )
    assert implementation is None and "ROCm device" in reason
    properties.assert_not_called()
    loader.assert_not_called()


def test_public_flash_attention_forwards_inputs_and_scale():
    query = torch.empty(1, 2, 8, 128)
    key = torch.empty(1, 3, 8, 128)
    value = torch.empty_like(key)
    expected = torch.empty_like(query)
    kernel = mock.Mock(return_value=expected)
    with mock.patch.object(
        gluon_dispatch, "get_flash_attention_impl", return_value=(kernel, None)
    ) as resolver:
        actual = gluon_dispatch.flash_attention(query, key, value, softmax_scale=0.07)
    assert actual is expected
    resolver.assert_called_once_with(query.device)
    kernel.assert_called_once_with(query, key, value, softmax_scale=0.07)


def test_public_flash_attention_reports_unavailable_implementation():
    query = torch.empty(1, 2, 8, 128)
    with (
        mock.patch.object(
            gluon_dispatch,
            "get_flash_attention_impl",
            return_value=(None, "GPU architecture gfx950"),
        ),
        pytest.raises(RuntimeError, match="GPU architecture gfx950"),
    ):
        gluon_dispatch.flash_attention(query, query, query)
