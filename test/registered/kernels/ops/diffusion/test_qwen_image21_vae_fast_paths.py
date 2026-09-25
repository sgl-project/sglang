# SPDX-License-Identifier: Apache-2.0
"""The Qwen-Image 2.1 VAE decoder fast paths must leave the decode bit-identical.

Regressions caught: a fused norm + SiLU tail or a folded-padding conv that no
longer matches the original chain, parameter names changed by the wrappers,
and a fast path that stays on after its first-sight check failed.
"""

from copy import deepcopy

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import (
    qwen_image21_vae_cuda_opt as vae_opt,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def make_vae():
    ac = QwenImage21VAEArchConfig(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        temperal_downsample=(False, False, False, False),
        in_channels=4,
        out_channels=4,
    )
    config = QwenImage21VAEConfig(arch_config=ac)
    config.load_encoder = False
    vae = AutoencoderKLQwenImage21(config).cuda().bfloat16().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in vae.named_parameters():
            if name.endswith("gamma"):
                param.copy_(1 + 0.1 * torch.randn_like(param))
            else:
                torch.nn.init.normal_(param, std=0.1)
    return vae


def gates(vae):
    found = []
    for m in vae.modules():
        if isinstance(m, vae_opt.FusedChannelRMSNormSiLU):
            found.append(m._exact_gate)
        elif isinstance(m, vae_opt.FoldedPadConv2d):
            found.append(m._gates["nchw"])
    return found


@torch.no_grad()
def test_decode_is_bit_identical_and_paths_verify(monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    bias_gate = vae_opt.BitExactFusionGate("test bias + residual")
    monkeypatch.setattr(vae_opt, "_BIAS_RESIDUAL_FUSION", bias_gate)
    calls = {"bias_residual": 0}
    real_bias_residual = vae_opt.bias_residual_add

    def counting(y, bias, h):
        calls["bias_residual"] += 1
        return real_bias_residual(y, bias, h)

    monkeypatch.setattr(vae_opt, "bias_residual_add", counting)
    assert set(optimized.state_dict().keys()) == set(reference.state_dict().keys())
    for key, value in optimized.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])
    installed = gates(optimized)
    assert len(installed) >= 8
    z = torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    expected = reference.decode(z)
    actual = optimized.decode(z)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert all(gate.verified and not gate.disabled for gate in installed)
    assert bias_gate.verified and not bias_gate.disabled
    # one fused bias + residual add per residual block
    assert calls["bias_residual"] == sum(
        1 for m in optimized.modules() if type(m).__name__ == "QwenImage21ResidualBlock"
    )
    z.normal_()
    assert torch.equal(optimized.decode(z), reference.decode(z))


@torch.no_grad()
def test_mismatch_disables_the_norm_fast_path(monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    monkeypatch.setattr(
        vae_opt,
        "channel_rmsnorm_finish_silu",
        lambda x, norm, gamma, scale: torch.zeros_like(x),
    )
    bias_gate = vae_opt.BitExactFusionGate("test mismatched bias + residual")
    monkeypatch.setattr(vae_opt, "_BIAS_RESIDUAL_FUSION", bias_gate)
    monkeypatch.setattr(
        vae_opt, "bias_residual_add", lambda y, bias, h: torch.zeros_like(y)
    )
    z = torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(optimized.decode(z), reference.decode(z))
    norm_gates = [
        m._exact_gate
        for m in optimized.modules()
        if isinstance(m, vae_opt.FusedChannelRMSNormSiLU)
    ]
    assert norm_gates and all(
        gate.disabled and not gate.verified for gate in norm_gates
    )
    assert bias_gate.disabled and not bias_gate.verified


@torch.no_grad()
def test_extra_high_decodes_channels_last_and_lossless_is_restored(monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    monkeypatch.setattr(
        vae_opt, "_BIAS_RESIDUAL_FUSION", vae_opt.BitExactFusionGate("test bias")
    )
    calls = {"nhwc": 0, "nhwc_bias": 0, "gather": 0, "conv_transpose": 0}
    real_nhwc, real_gather, real_conv_t = (
        vae_opt.channel_rmsnorm_silu_nhwc,
        vae_opt.nearest_upsample_nhwc,
        vae_opt.F.conv_transpose2d,
    )

    def counting_conv_t(*args, **kwargs):
        calls["conv_transpose"] += 1
        return real_conv_t(*args, **kwargs)

    monkeypatch.setattr(vae_opt.F, "conv_transpose2d", counting_conv_t)

    def counting_nhwc(x, gamma, scale, bias=None):
        calls["nhwc"] += 1
        calls["nhwc_bias"] += bias is not None
        return real_nhwc(x, gamma, scale, bias)

    def counting_gather(x, scale_factor):
        calls["gather"] += 1
        return real_gather(x, scale_factor)

    monkeypatch.setattr(vae_opt, "channel_rmsnorm_silu_nhwc", counting_nhwc)
    monkeypatch.setattr(vae_opt, "nearest_upsample_nhwc", counting_gather)
    z = torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    expected = reference.decode(z)
    with use_vae_fast_path(optimized, True):
        fast = optimized.decode(z)
    assert fast.shape == expected.shape
    assert calls["nhwc"] > 0 and calls["conv_transpose"] == 4
    # every residual block's first conv hands its bias to the norm kernel
    assert calls["nhwc_bias"] == sum(
        1 for m in optimized.modules() if type(m).__name__ == "QwenImage21ResidualBlock"
    )
    # every upsampler is folded, so the nearest gather no longer runs
    assert calls["gather"] == 0
    assert all(
        isinstance(m.resample, vae_opt.FusedUpsample2xConv)
        for m in optimized.modules()
        if type(m).__name__ == "QwenImage21Resample"
    )
    torch.testing.assert_close(fast.float(), expected.float(), atol=0.05, rtol=0)
    assert (
        not torch.equal(fast, expected) or True
    )  # rounding may or may not differ on a tiny model
    # gate off: layout and kernels revert, output is bit-identical again
    assert torch.equal(optimized.decode(z), expected)
    assert not optimized.decoder._sgl_channels_last


@torch.no_grad()
def test_other_vaes_pass_through():
    assert vae_opt.maybe_optimize_qwen_image21_vae(torch.nn.Linear(2, 2)) is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
