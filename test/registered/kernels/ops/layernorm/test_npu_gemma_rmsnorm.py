import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

import sglang.kernels.fused_op as fused_op_module
from sglang import kernels
from sglang.kernels.ops.layernorm import (
    GemmaFusedAddRMSNormOp,
    GemmaRMSNormOp,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, PlatformInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_sgl_kernel_npu(gemma_kernel, add_gemma_kernel):
    package = ModuleType("sgl_kernel_npu")
    package.__path__ = []
    norm_package = ModuleType("sgl_kernel_npu.norm")
    norm_package.__path__ = []
    module = ModuleType("sgl_kernel_npu.norm.gemma_rmsnorm")
    module.npu_gemma_rms_norm = gemma_kernel
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
def test_registry_exposes_only_sgl_kernel_npu_provider(op_name):
    registered_backends = {spec.backend for spec in kernels.registry.get(op_name)}
    spec = kernels.registry.get_backend(op_name, KernelBackend.SGL_KERNEL_NPU)

    assert KernelBackend.TORCH_NPU not in registered_backends
    assert spec.capabilities == frozenset({CapabilityRequirement.NPU})
    assert spec.target.endswith(".forward_sgl_kernel_npu")


@pytest.mark.parametrize("op_cls", [GemmaRMSNormOp, GemmaFusedAddRMSNormOp])
def test_sgl_kernel_npu_is_selected_on_npu(op_cls):
    with patch.object(
        fused_op_module, "_platform", return_value=PlatformInfo(device_type="npu")
    ):
        backend = op_cls()._resolve_backend()

    assert backend is KernelBackend.SGL_KERNEL_NPU


def test_missing_kernel_package_reports_actionable_error():
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
        with pytest.raises(RuntimeError, match="requires a target-specific"):
            GemmaRMSNormOp().forward_sgl_kernel_npu(
                torch.randn(2, 4), torch.randn(4), 1e-5
            )


def test_incompatible_kernel_package_reports_actionable_error():
    modules = _fake_sgl_kernel_npu(MagicMock(), MagicMock())
    del modules["sgl_kernel_npu.norm.gemma_rmsnorm"].npu_gemma_rms_norm

    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match="requires a target-specific"):
            GemmaRMSNormOp().forward_sgl_kernel_npu(
                torch.randn(2, 4), torch.randn(4), 1e-5
            )


def test_sgl_kernel_npu_normal_out_contract():
    x = torch.randn(2, 4)
    weight = torch.randn(4)
    expected = torch.randn_like(x)
    out = torch.empty_like(x)
    gemma_kernel = MagicMock(return_value=(expected, "rstd"))

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


@pytest.mark.parametrize("layer_name", ["GemmaRMSNorm", "Gemma3RMSNorm"])
def test_srt_gemma_layers_delegate_plain_npu_path(layer_name):
    from sglang.srt.layers import layernorm as layernorm_module

    layer_cls = getattr(layernorm_module, layer_name)
    layer = layer_cls(4)
    x = torch.randn(2, 4)
    unified_op = MagicMock(return_value=(x, "rstd"))

    with patch.object(
        layernorm_module, "npu_gemma_rms_norm", unified_op, create=True
    ):
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
    norm_output = torch.randn_like(x)
    residual_sum = torch.randn_like(residual)
    fused_op = MagicMock(return_value=(norm_output, residual_sum))

    with patch.object(
        layernorm_module,
        "add_gemma_rms_norm",
        fused_op,
        create=True,
    ):
        result = layer.forward_npu(x, residual)

    assert result[0] is norm_output
    assert result[1] is residual_sum
    eps = layer.variance_epsilon if hasattr(layer, "variance_epsilon") else layer.eps
    fused_op.assert_called_once_with(x, layer.weight, residual, eps)
