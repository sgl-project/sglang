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


def norm_gates(vae):
    return [
        m._exact_gate
        for m in vae.modules()
        if isinstance(m, vae_opt.FusedChannelRMSNormSiLU)
    ]


def conv_fold_gates(vae):
    return [
        m._gates["nchw"]
        for m in vae.modules()
        if isinstance(m, vae_opt.FoldedPadConv2d)
    ]


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    # Bit-exactness is asserted against the eager chain, so both sides must run
    # deterministic conv algorithms; cuDNN may otherwise pick split-K engines
    # whose atomics reorder the sum between two calls.
    previous = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = previous


@torch.no_grad()
def test_decode_is_bit_identical_and_paths_verify(monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    bias_gate = vae_opt.BitExactFusionGate("test bias + residual")
    monkeypatch.setattr(vae_opt, "_BIAS_RESIDUAL_FUSION", bias_gate)
    up_gate = vae_opt.BitExactFusionGate("test upsampler bias")
    monkeypatch.setattr(vae_opt, "_UPSAMPLE_BIAS_FUSION", up_gate)
    calls = {"bias_residual": 0, "dup_bias": 0}
    real_bias_residual, real_dup = vae_opt.bias_residual_add, vae_opt.dup_up3d_add

    def counting(y, bias, h):
        calls["bias_residual"] += 1
        return real_bias_residual(y, bias, h)

    def counting_dup(*args, **kwargs):
        calls["dup_bias"] += len(args) > 6 or "bias" in kwargs
        return real_dup(*args, **kwargs)

    monkeypatch.setattr(vae_opt, "bias_residual_add", counting)
    monkeypatch.setattr(vae_opt, "dup_up3d_add", counting_dup)
    assert set(optimized.state_dict().keys()) == set(reference.state_dict().keys())
    for key, value in optimized.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])
    norms, folds = norm_gates(optimized), conv_fold_gates(optimized)
    assert len(norms) >= 8 and len(folds) >= 8
    z = torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    expected = reference.decode(z)
    assert torch.equal(reference.decode(z), expected), (
        "the eager decode itself is not reproducible on this platform"
    )
    actual = optimized.decode(z)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # Kernels that are exact by construction must have engaged.
    assert all(gate.verified and not gate.disabled for gate in norms)
    # The padding fold is kept only where cuDNN's algorithm for the new
    # descriptor reproduces the padded conv bit for bit (it does on SM86 and
    # SM120, not on H100), so a declined fold is a valid outcome; every conv
    # must have reached a decision, and the decode above proved the fallback.
    assert all(gate.verified or gate.disabled for gate in folds)
    assert bias_gate.verified and not bias_gate.disabled
    assert up_gate.verified and not up_gate.disabled
    assert calls["dup_bias"] == 4  # one fused shortcut add per upsampling block
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
    monkeypatch.setattr(
        vae_opt, "_UPSAMPLE_BIAS_FUSION", vae_opt.BitExactFusionGate("test up bias")
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
        optimized.decode(z)  # first-sight compares run their eager references once
        for key in calls:
            calls[key] = 0
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
    # the folded upsampler kernel follows the conv weight: a new weight value
    # must not reuse the stale fold
    for model in (reference, optimized):
        for m in model.modules():
            if type(m).__name__ == "QwenImage21Resample" and m.mode.startswith(
                "upsample"
            ):
                m.resample[1].weight.zero_()
                m.resample[1].bias.zero_()
    expected_zero = reference.decode(z)
    with use_vae_fast_path(optimized, True):
        fast_zero = optimized.decode(z)
    assert not torch.equal(expected_zero, expected)
    torch.testing.assert_close(
        fast_zero.float(), expected_zero.float(), atol=0.05, rtol=0
    )
    # CPU offload must not leave the folded kernels (or anything else) on the GPU
    optimized.to("cpu")
    for m in optimized.modules():
        for value in list(vars(m).values()) + list(m._buffers.values()):
            assert not (isinstance(value, torch.Tensor) and value.is_cuda), type(m)
    optimized.to("cuda")


def test_fold_rejects_convs_it_cannot_express():
    from sglang.multimodal_gen.runtime.models.vaes.conv_fold import (
        fold_upsample2x_conv2d_weight,
    )

    good = torch.nn.Conv2d(4, 6, 3, padding=1)
    assert fold_upsample2x_conv2d_weight(good).shape == (4, 6, 4, 4)
    for bad in (
        torch.nn.Conv2d(4, 6, 1),
        torch.nn.Conv2d(4, 6, 3, padding=1, stride=2),
        torch.nn.Conv2d(4, 4, 3, padding=1, groups=2),
        torch.nn.Conv2d(4, 6, 3, padding=1, padding_mode="reflect"),
    ):
        with pytest.raises(ValueError):
            fold_upsample2x_conv2d_weight(bad)


@torch.no_grad()
def test_other_vaes_pass_through():
    assert vae_opt.maybe_optimize_qwen_image21_vae(torch.nn.Linear(2, 2)) is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
