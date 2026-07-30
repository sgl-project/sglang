import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import sglang.kernels.fused_op as fused_op_module
from sglang import kernels
from sglang.kernels.ops.layernorm import (
    GemmaFusedAddRMSNormOp,
    GemmaRMSNormOp,
    _load_sgl_kernel_npu_gemma_api,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, PlatformInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def clear_gemma_api_cache():
    _load_sgl_kernel_npu_gemma_api.cache_clear()
    yield
    _load_sgl_kernel_npu_gemma_api.cache_clear()


def _fake_sgl_kernel_npu(gemma_kernel, add_gemma_kernel):
    package = ModuleType("sgl_kernel_npu")
    package.__path__ = []
    norm_package = ModuleType("sgl_kernel_npu.norm")
    norm_package.__path__ = []
    module = ModuleType("sgl_kernel_npu.norm.gemma_rmsnorm")
    module.gemma_rms_norm = gemma_kernel
    module.add_gemma_rms_norm = add_gemma_kernel
    return {
        "sgl_kernel_npu": package,
        "sgl_kernel_npu.norm": norm_package,
        "sgl_kernel_npu.norm.gemma_rmsnorm": module,
    }


@pytest.mark.parametrize(
    "op_name",
    ["layernorm.gemma_rmsnorm", "layernorm.gemma_fused_add_rmsnorm"],
)
def test_registry_exposes_kernel_and_native_npu_providers(op_name):
    registered_backends = {spec.backend for spec in kernels.registry.get(op_name)}
    spec = kernels.registry.get_backend(op_name, KernelBackend.SGL_KERNEL_NPU)
    native_spec = kernels.registry.get_backend(op_name, KernelBackend.TORCH_NPU)

    assert KernelBackend.TORCH_NPU in registered_backends
    assert spec.capabilities == frozenset({CapabilityRequirement.NPU})
    assert native_spec.capabilities == frozenset({CapabilityRequirement.NPU})
    assert spec.target.endswith(".forward_sgl_kernel_npu")
    assert native_spec.target.endswith(".forward_npu")


@pytest.mark.parametrize("op_cls", [GemmaRMSNormOp, GemmaFusedAddRMSNormOp])
def test_sgl_kernel_npu_is_selected_on_npu(op_cls):
    with (
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
        ),
        patch(
            "sglang.kernels.ops.layernorm._load_sgl_kernel_npu_gemma_api",
            return_value=(MagicMock(), MagicMock()),
        ),
    ):
        backend = op_cls()._resolve_backend()

    assert backend is KernelBackend.SGL_KERNEL_NPU


@pytest.mark.parametrize("op_cls", [GemmaRMSNormOp, GemmaFusedAddRMSNormOp])
def test_torch_npu_is_selected_when_target_wheel_has_no_gemma_api(op_cls):
    with (
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
        ),
        patch(
            "sglang.kernels.ops.layernorm._load_sgl_kernel_npu_gemma_api",
            return_value=None,
        ),
    ):
        backend = op_cls()._resolve_backend()

    assert backend is KernelBackend.TORCH_NPU


def test_loader_reports_missing_target_specific_api():
    package = ModuleType("sgl_kernel_npu")
    package.__path__ = []
    norm_package = ModuleType("sgl_kernel_npu.norm")
    norm_package.__path__ = []

    with patch.dict(
        sys.modules,
        {
            "sgl_kernel_npu": package,
            "sgl_kernel_npu.norm": norm_package,
        },
    ):
        sys.modules.pop("sgl_kernel_npu.norm.gemma_rmsnorm", None)
        assert _load_sgl_kernel_npu_gemma_api() is None


def test_sgl_kernel_npu_selection_does_not_query_soc():
    get_soc_version = MagicMock(side_effect=AssertionError("SoC query is forbidden"))
    torch_npu = SimpleNamespace(npu=SimpleNamespace(get_soc_version=get_soc_version))
    op = GemmaRMSNormOp()

    with (
        patch.dict(sys.modules, {"torch_npu": torch_npu}),
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
        ),
        patch(
            "sglang.kernels.ops.layernorm._load_sgl_kernel_npu_gemma_api",
            return_value=(MagicMock(), MagicMock()),
        ),
    ):
        assert op._resolve_backend(torch.randn(2, 4), torch.randn(4)) is (
            KernelBackend.SGL_KERNEL_NPU
        )

    get_soc_version.assert_not_called()


def test_sgl_kernel_npu_normal_out_contract():
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    expected = torch.randn_like(x)
    out = torch.empty_like(x)
    gemma_kernel = MagicMock(return_value=expected)

    with patch.dict(sys.modules, _fake_sgl_kernel_npu(gemma_kernel, MagicMock())):
        result = GemmaRMSNormOp().forward_sgl_kernel_npu(x, weight, 1e-5, out=out)

    assert result is out
    torch.testing.assert_close(out, expected)
    gemma_kernel.assert_called_once_with(x, weight, 1e-5)


def test_sgl_kernel_npu_fused_in_place_contract():
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.randn(4)
    norm_output = torch.randn_like(x)
    residual_sum = torch.randn_like(residual)
    add_gemma_kernel = MagicMock(return_value=(norm_output, residual_sum))

    with patch.dict(sys.modules, _fake_sgl_kernel_npu(MagicMock(), add_gemma_kernel)):
        result = GemmaFusedAddRMSNormOp().forward_sgl_kernel_npu(
            x, residual, weight, 1e-5
        )

    assert result is None
    torch.testing.assert_close(x, norm_output)
    torch.testing.assert_close(residual, residual_sum)
    add_gemma_kernel.assert_called_once()
    args = add_gemma_kernel.call_args.args
    assert args[0] is x
    assert args[1] is weight
    assert args[2] is residual
    assert args[3] == 1e-5


def test_torch_npu_normal_out_contract():
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    expected = torch.randn_like(x)
    out = torch.empty_like(x)
    native_kernel = MagicMock(return_value=(expected, None))

    with patch.dict(
        sys.modules,
        {"torch_npu": SimpleNamespace(npu_gemma_rms_norm=native_kernel)},
    ):
        result = GemmaRMSNormOp().forward_npu(x, weight, 1e-5, out=out)

    assert result is out
    torch.testing.assert_close(out, expected)
    native_kernel.assert_called_once_with(x, weight, 1e-5)


def test_torch_npu_fused_in_place_contract():
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.randn(4)
    norm_output = torch.randn_like(x)
    residual_sum = torch.randn_like(residual)
    native_kernel = MagicMock(return_value=(norm_output, None, residual_sum))

    with patch.dict(
        sys.modules,
        {"torch_npu": SimpleNamespace(npu_add_rms_norm=native_kernel)},
    ):
        result = GemmaFusedAddRMSNormOp().forward_npu(x, residual, weight, 1e-5)

    assert result is None
    torch.testing.assert_close(x, norm_output)
    torch.testing.assert_close(residual, residual_sum)
    native_kernel.assert_called_once()
    args = native_kernel.call_args.args
    assert args[0] is residual
    assert args[1] is x
    torch.testing.assert_close(args[2], 1.0 + weight)
    assert args[3] == 1e-5


@pytest.mark.parametrize("layer_name", ["GemmaRMSNorm", "Gemma3RMSNorm"])
def test_srt_gemma_layers_delegate_plain_npu_path(layer_name):
    from sglang.srt.layers import layernorm as layernorm_module

    layer_cls = getattr(layernorm_module, layer_name)
    layer = layer_cls(4)
    x = torch.randn(2, 4)
    unified_op = MagicMock(return_value=x)

    with patch.object(layernorm_module, "npu_gemma_rmsnorm", unified_op, create=True):
        result = layer.forward_npu(x)

    assert result is x
    eps = layer.variance_epsilon if hasattr(layer, "variance_epsilon") else layer.eps
    unified_op.assert_called_once_with(x, layer.weight, eps)


@pytest.mark.parametrize("layer_name", ["GemmaRMSNorm", "Gemma3RMSNorm"])
def test_srt_gemma_layers_delegate_residual_npu_path(layer_name):
    from sglang.srt.layers import layernorm as layernorm_module

    layer_cls = getattr(layernorm_module, layer_name)
    layer = layer_cls(4)
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    fused_op = MagicMock()

    with patch.object(
        layernorm_module,
        "npu_gemma_fused_add_rmsnorm",
        fused_op,
        create=True,
    ):
        result = layer.forward_npu(x, residual)

    assert result[0] is x
    assert result[1] is residual
    eps = layer.variance_epsilon if hasattr(layer, "variance_epsilon") else layer.eps
    fused_op.assert_called_once_with(x, residual, layer.weight, eps)
