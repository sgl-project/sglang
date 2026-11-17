"""SM120-only model fast paths split out of ``test_model_fast_paths.py``.

SANA-Video's depthwise GLU and Joy's image MLP epilogue are enabled on SM120
only, so they run on the RTX 5090 lane; the other model wrappers run on H100.
"""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.sana as sana
from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (12, 0),
    reason="these fast paths are enabled on SM120 only",
)


# -------------------------------------------------------------------------
# SANA-Video -- depthwise convolution and rounded GLU
# -------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 34, 1, 1), (2, 192, 7, 11), (21, 13440, 30, 52)])
@torch.no_grad()
def test_sana_video_depthwise_glu_rounding_and_row_tails(shape):
    """Keep each BF16 rounding boundary and mask partial four-row tiles."""
    from sglang.kernels.ops.diffusion import fused_depthwise_conv3x3_bias_glu

    torch.manual_seed(606)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    weight = torch.randn((shape[1], 1, 3, 3), device=x.device, dtype=x.dtype) * 0.1
    bias = torch.randn(shape[1], device=x.device, dtype=x.dtype) * 0.1
    raw = F.conv2d(x, weight, None, padding=1, groups=shape[1])
    hidden, gate = (raw + bias[None, :, None, None]).chunk(2, dim=1)
    actual = fused_depthwise_conv3x3_bias_glu(x, weight, bias)
    assert torch.equal(actual, hidden * F.silu(gate))
    assert actual.is_contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def test_sana_video_depthwise_gate_capture_and_changed_inputs(monkeypatch):
    """Capture cannot verify; warmed replay must read current inputs and bias."""
    from sglang.multimodal_gen.runtime.models.dits import sana_video

    gate = BitExactFusionGate("test depthwise", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    torch.manual_seed(606)
    module = sana_video.GLUMBTempConv(32, 3).cuda().bfloat16()
    module.conv_depth = (
        nn.Conv2d(13440, 13440, 3, padding=1, groups=13440).cuda().bfloat16()
    )
    x = torch.randn((2, 13440, 30, 52), device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    expected = sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unverified = module._depthwise_glu(x)
    graph.replay()
    assert torch.equal(unverified, expected)
    assert not gate.verified and not gate.disabled
    assert torch.equal(module._depthwise_glu(x), expected)
    assert gate.verified and not gate.disabled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = module._depthwise_glu(x)
    x.add_(0.125)
    module.conv_depth.bias.mul_(0.75)
    graph.replay()
    assert torch.equal(
        actual, sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    )
    gate = BitExactFusionGate("test mismatch", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    expected = sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    with monkeypatch.context() as context:
        context.setattr(
            sana_video,
            "fused_depthwise_conv3x3_bias_glu",
            lambda *args: torch.zeros_like(expected),
        )
        assert torch.equal(module._depthwise_glu(x), expected)
    assert gate.disabled
    x.mul_(0.5)
    assert torch.equal(
        module._depthwise_glu(x),
        sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True),
    )


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_sana_video_depthwise_reference_fallback(monkeypatch, grad_enabled):
    from sglang.multimodal_gen.runtime.models.dits import sana_video

    gate = BitExactFusionGate("test fallback", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    torch.manual_seed(606)
    module = sana_video.GLUMBTempConv(32, 3).cuda().bfloat16()
    # Both shapes pass the size floor. Use two groups for the non-cuDNN path.
    shape = (2, 13440, 30, 52) if grad_enabled else (1, 2, 4096, 2048)
    module.conv_depth = (
        nn.Conv2d(shape[1], shape[1], 3, padding=1, groups=shape[1]).cuda().bfloat16()
    )
    x = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        .to(memory_format=torch.channels_last)
        .requires_grad_(grad_enabled)
    )
    with (
        torch.set_grad_enabled(grad_enabled),
        torch.backends.cudnn.flags(enabled=grad_enabled),
    ):
        actual = module._depthwise_glu(x)
        expected = sana.sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
        assert torch.equal(actual, expected)
        if grad_enabled:
            actual_grad = torch.autograd.grad(actual.sum(), x)[0]
            expected_grad = torch.autograd.grad(expected.sum(), x)[0]
            assert torch.equal(actual_grad, expected_grad)
    assert not gate.verified and not gate.disabled


# -------------------------------------------------------------------------
# Joy -- high-mode image MLP epilogue on SM120
# -------------------------------------------------------------------------


@torch.no_grad()
def test_joy_image_mlp_quality_restore_and_dynamic_lora():
    from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer
    from sglang.multimodal_gen.runtime.layers.mlp import MLP
    from sglang.multimodal_gen.runtime.models.dits import joy_image

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(42)
    mlp = MLP(4096, 16384).to(device="cuda", dtype=torch.bfloat16)
    for parameter in mlp.parameters():
        parameter.normal_(0, 0.01)
    x = torch.randn(1, 7168, 4096, device="cuda", dtype=torch.bfloat16)
    reference = mlp(x)
    with patch.object(
        joy_image, "fused_linear_gelu_tanh", wraps=joy_image.fused_linear_gelu_tanh
    ) as fused:
        torch.testing.assert_close(
            joy_image._joy_image_mlp(mlp, x, fuse_gelu=True),
            reference,
            atol=2e-2,
            rtol=2e-2,
        )
        assert fused.call_count == 1
        assert torch.equal(joy_image._joy_image_mlp(mlp, x, fuse_gelu=False), reference)
        # Smaller text/token counts must keep their original, faster projection.
        text = x[:, :1053].contiguous()
        assert torch.equal(
            joy_image._joy_image_mlp(mlp, text, fuse_gelu=True), mlp(text)
        )
        assert fused.call_count == 1

        compiled = torch.compile(
            joy_image._joy_image_mlp, backend="eager", fullgraph=True
        )
        assert torch.equal(compiled(mlp, x, True), reference)
        assert fused.call_count == 1

        wrapper = wrap_with_lora_layer(mlp.fc_in, lora_rank=4, lora_alpha=4)
        wrapper.set_lora_weights(
            torch.randn(4, 4096, device=x.device, dtype=x.dtype) * 0.02,
            torch.randn(16384, 4, device=x.device, dtype=x.dtype) * 0.02,
            merge_weights=False,
        )
        mlp.fc_in = wrapper
        adapted = mlp(x)
        assert not torch.equal(adapted, reference)
        assert torch.equal(joy_image._joy_image_mlp(mlp, x, fuse_gelu=True), adapted)
        assert fused.call_count == 1
        wrapper.disable_lora = True
        joy_image._joy_image_mlp(mlp, x, fuse_gelu=True)
        assert fused.call_count == 2
        assert torch.equal(joy_image._joy_image_mlp(mlp, x, fuse_gelu=False), reference)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
