import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


@pytest.mark.parametrize("hidden_size", [256, 5120, 8192])
@pytest.mark.parametrize(
    ("weight_dtype", "input_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("with_post_residual", [False, True])
def test_srt_gemma_residual_reuses_precomputed_weight(
    hidden_size, weight_dtype, input_dtype, with_post_residual
):
    from sglang.srt.layers import layernorm as layernorm_module

    layer = layernorm_module.GemmaRMSNorm(hidden_size).to(weight_dtype)
    weight = torch.linspace(-0.5, 0.5, hidden_size, dtype=weight_dtype)
    layer.weight.weight_loader(layer.weight, weight)
    x = torch.randn(2, hidden_size, dtype=input_dtype)
    residual = torch.randn_like(x)
    post_residual = torch.randn_like(x) if with_post_residual else None
    output, residual_sum = torch.empty_like(x), torch.empty_like(x)
    fused_op = MagicMock(return_value=(output, None, residual_sum))

    with patch.object(
        layernorm_module,
        "torch_npu",
        SimpleNamespace(npu_add_rms_norm=fused_op),
        create=True,
    ):
        result = layer.forward_npu(x, residual, post_residual)

    assert result[0] is output
    assert result[1] is residual_sum
    fused_op.assert_called_once()
    actual_residual, actual_x, gamma, eps = fused_op.call_args.args
    assert actual_x is x
    assert eps == layer.variance_epsilon
    if with_post_residual:
        torch.testing.assert_close(actual_residual, residual + post_residual)
    else:
        assert actual_residual is residual
    assert gamma.dtype == input_dtype
    torch.testing.assert_close(gamma, (1.0 + weight).to(input_dtype))
    if weight_dtype == input_dtype:
        assert gamma is layer.gemma_weight


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
