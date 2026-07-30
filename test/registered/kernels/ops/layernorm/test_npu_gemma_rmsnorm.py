import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import sglang.kernels as kernels
import sglang.kernels.fused_op as fused_op_module
import sglang.kernels.ops.layernorm as unified_layernorm
from sglang.kernels.fused_op import BACKEND_METHODS
from sglang.kernels.ops.layernorm import (
    GemmaFusedAddRMSNormOp,
    GemmaRMSNormOp,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, PlatformInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _reset_fused_op_state():
    unified_layernorm._load_sgl_kernel_npu_gemma_ops.cache_clear()
    yield
    unified_layernorm._load_sgl_kernel_npu_gemma_ops.cache_clear()
    kernels.set_fused_op_backend(None)
    kernels.disable_fused_op_trace()
    kernels.clear_fused_op_trace()


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


def _fake_old_sgl_kernel_npu():
    package = ModuleType("sgl_kernel_npu")
    package.__path__ = []
    norm_package = ModuleType("sgl_kernel_npu.norm")
    norm_package.__path__ = []
    return {
        "sgl_kernel_npu": package,
        "sgl_kernel_npu.norm": norm_package,
    }


def test_backend_enum_method_mapping_and_priority():
    assert KernelBackend.SGL_KERNEL_NPU.value == "sgl_kernel_npu"
    assert BACKEND_METHODS[KernelBackend.SGL_KERNEL_NPU] == "forward_sgl_kernel_npu"
    assert unified_layernorm._NORM_PRIORITY.index(
        KernelBackend.SGL_KERNEL_NPU
    ) < unified_layernorm._NORM_PRIORITY.index(KernelBackend.TORCH_NPU)


@pytest.mark.parametrize(
    "op_name",
    ["layernorm.gemma_rmsnorm", "layernorm.gemma_fused_add_rmsnorm"],
)
def test_registry_exposes_sgl_kernel_npu_with_npu_capability(op_name):
    spec = kernels.registry.get_backend(op_name, KernelBackend.SGL_KERNEL_NPU)

    assert spec.capabilities == frozenset({CapabilityRequirement.NPU})
    assert spec.target.endswith(".forward_sgl_kernel_npu")


def test_lazy_loader_finds_stable_kernel_api():
    gemma_kernel = MagicMock()
    add_gemma_kernel = MagicMock()

    with patch.dict(sys.modules, _fake_sgl_kernel_npu(gemma_kernel, add_gemma_kernel)):
        ops = unified_layernorm._load_sgl_kernel_npu_gemma_ops()

    assert ops == (gemma_kernel, add_gemma_kernel)


def test_lazy_loader_rejects_old_package_without_stable_api():
    with patch.dict(sys.modules, _fake_old_sgl_kernel_npu()):
        assert unified_layernorm._load_sgl_kernel_npu_gemma_ops() is None


def test_new_kernel_api_is_preferred_on_npu():
    op = GemmaRMSNormOp()
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    ops = (MagicMock(), MagicMock())

    with (
        patch.object(
            unified_layernorm, "_load_sgl_kernel_npu_gemma_ops", return_value=ops
        ),
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
        ),
    ):
        backend = op._resolve_backend(x, weight)

    assert backend is KernelBackend.SGL_KERNEL_NPU


def test_old_kernel_package_falls_back_to_torch_npu():
    op = GemmaRMSNormOp()
    x = torch.randn(2, 4)
    weight = torch.randn(4)

    with (
        patch.object(
            unified_layernorm, "_load_sgl_kernel_npu_gemma_ops", return_value=None
        ),
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
        ),
    ):
        backend = op._resolve_backend(x, weight)

    assert backend is KernelBackend.TORCH_NPU


def test_non_npu_selection_does_not_import_sgl_kernel_npu():
    op = GemmaRMSNormOp()

    with (
        patch.object(
            unified_layernorm,
            "_load_sgl_kernel_npu_gemma_ops",
            side_effect=AssertionError("NPU package import is forbidden"),
        ),
        patch.object(
            fused_op_module,
            "_platform",
            return_value=PlatformInfo(device_type="cpu"),
        ),
    ):
        assert not op.backend_eligible(KernelBackend.SGL_KERNEL_NPU)


def test_sgl_kernel_npu_selection_does_not_query_soc():
    gemma_kernel = MagicMock()
    add_gemma_kernel = MagicMock()
    get_soc_version = MagicMock(side_effect=AssertionError("SoC query is forbidden"))
    torch_npu = SimpleNamespace(npu=SimpleNamespace(get_soc_version=get_soc_version))
    op = GemmaRMSNormOp()

    with (
        patch.dict(
            sys.modules,
            {
                **_fake_sgl_kernel_npu(gemma_kernel, add_gemma_kernel),
                "torch_npu": torch_npu,
            },
        ),
        patch.object(
            fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
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

    with patch.object(
        unified_layernorm,
        "_load_sgl_kernel_npu_gemma_ops",
        return_value=(gemma_kernel, MagicMock()),
    ):
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

    with patch.object(
        unified_layernorm,
        "_load_sgl_kernel_npu_gemma_ops",
        return_value=(MagicMock(), add_gemma_kernel),
    ):
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


def test_torch_npu_normal_fallback_uses_offset_weight():
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    fallback_kernel = MagicMock(return_value=(x, None))
    torch_npu = SimpleNamespace(npu_rms_norm=fallback_kernel)

    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        result = GemmaRMSNormOp().forward_npu(x, weight)

    assert result is x
    args = fallback_kernel.call_args.args
    assert args[0] is x
    torch.testing.assert_close(args[1], 1.0 + weight)
    assert args[2] == 1e-6


def test_torch_npu_fused_fallback_uses_offset_weight_and_writes_back():
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.randn(4)
    norm_output = torch.randn_like(x)
    residual_sum = torch.randn_like(residual)
    fallback_kernel = MagicMock(return_value=(norm_output, None, residual_sum))
    torch_npu = SimpleNamespace(npu_add_rms_norm=fallback_kernel)

    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        result = GemmaFusedAddRMSNormOp().forward_npu(x, residual, weight)

    assert result is None
    torch.testing.assert_close(x, norm_output)
    torch.testing.assert_close(residual, residual_sum)
    args = fallback_kernel.call_args.args
    assert args[0] is residual
    assert args[1] is x
    torch.testing.assert_close(args[2], 1.0 + weight)
    assert args[3] == 1e-6


def test_force_backend_and_trace_use_sgl_kernel_npu():
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    gemma_kernel = MagicMock(return_value=x)
    kernels.set_fused_op_backend(KernelBackend.SGL_KERNEL_NPU)
    kernels.enable_fused_op_trace()

    with patch.object(
        unified_layernorm,
        "_load_sgl_kernel_npu_gemma_ops",
        return_value=(gemma_kernel, MagicMock()),
    ):
        result = unified_layernorm.gemma_rmsnorm(x, weight)

    assert result is x
    (record,) = kernels.get_fused_op_trace()
    assert record.op == "layernorm.gemma_rmsnorm"
    assert record.backend == "sgl_kernel_npu"


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
