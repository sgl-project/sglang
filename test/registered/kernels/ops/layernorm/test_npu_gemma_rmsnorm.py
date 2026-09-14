import re
import sys
from pathlib import Path
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
    gemma_module = ModuleType("sgl_kernel_npu.norm.gemma_rmsnorm")
    gemma_module.npu_gemma_rms_norm = gemma_kernel
    add_module = ModuleType("sgl_kernel_npu.norm.add_rmsnorm_bias")
    add_module.add_gemma_rms_norm = add_gemma_kernel
    return {
        "sgl_kernel_npu": package,
        "sgl_kernel_npu.norm": norm_package,
        "sgl_kernel_npu.norm.gemma_rmsnorm": gemma_module,
        "sgl_kernel_npu.norm.add_rmsnorm_bias": add_module,
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
        backend = op_cls().auto_selected_backend()

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
    """A wheel that imports but lacks the symbol must name the module it lacks.

    Covers the AttributeError branch (the test above covers ImportError), and
    pins each op to its own module so the message cannot drift into a
    copy-pasted sibling name.
    """
    modules = _fake_sgl_kernel_npu(MagicMock(), MagicMock())
    del modules["sgl_kernel_npu.norm.gemma_rmsnorm"].npu_gemma_rms_norm
    del modules["sgl_kernel_npu.norm.add_rmsnorm_bias"].add_gemma_rms_norm

    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match=r"norm\.gemma_rmsnorm\."):
            GemmaRMSNormOp().forward_sgl_kernel_npu(
                torch.randn(2, 4), torch.randn(4), 1e-5
            )
        with pytest.raises(RuntimeError, match=r"norm\.add_rmsnorm_bias\."):
            GemmaFusedAddRMSNormOp().forward_sgl_kernel_npu(
                torch.randn(2, 4), torch.randn(2, 4), torch.randn(4), 1e-5
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


@pytest.mark.parametrize(
    ("layer_name", "eps_attr"),
    [("GemmaRMSNorm", "variance_epsilon"), ("Gemma3RMSNorm", "eps")],
)
def test_srt_gemma_layers_delegate_plain_npu_path(layer_name, eps_attr):
    from sglang.srt.layers import layernorm as layernorm_module

    layer = getattr(layernorm_module, layer_name)(4)
    x = torch.randn(2, 4)
    unified_op = MagicMock(return_value=(x, "rstd"))

    with patch.object(layernorm_module, "npu_gemma_rms_norm", unified_op, create=True):
        result = layer.forward_npu(x)

    assert result is x
    unified_op.assert_called_once_with(x, layer.weight, getattr(layer, eps_attr))


def test_srt_falls_back_to_torch_npu_on_wheels_without_the_provider():
    """The srt-layer provider import must stay guarded, with a torch_npu fallback.

    ``srt/layers/layernorm.py`` imports the provider at module scope under
    ``if _is_npu``, which CPU CI never executes -- hence a source-shape check.
    Turning it back into a hard import would break every NPU deployment running
    an sgl-kernel-npu wheel from before the staged Gemma provider, at
    ``import sglang.srt.layers.layernorm`` time and for all models, not just
    Gemma ones.
    """
    from sglang.srt.layers import layernorm as layernorm_module

    source = Path(layernorm_module.__file__).read_text(encoding="utf-8")
    guarded_import = re.search(
        r"try:\s*\n"
        r"\s*from sgl_kernel_npu\.norm\.gemma_rmsnorm import npu_gemma_rms_norm\s*\n"
        r"\s*except ImportError:\s*\n"
        r"(?:\s*#.*\n)*"
        r"\s*npu_gemma_rms_norm = torch_npu\.npu_gemma_rms_norm\s*\n",
        source,
    )

    assert guarded_import is not None
