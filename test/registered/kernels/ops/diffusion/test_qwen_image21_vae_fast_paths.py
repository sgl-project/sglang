# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE decoder fast paths.

The default (lossless) paths must leave the decode bit-identical to the original
module chain and fall back when a first-sight check fails; the quality-gated
channels_last paths must stay close to it and switch off cleanly.
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
    QwenImage21Resample,
)
from sglang.multimodal_gen.runtime.models.vaes.conv_fold import (
    fold_upsample2x_conv2d_weight,
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


def latent():
    return torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)


def modules_of(vae, cls):
    return [m for m in vae.modules() if isinstance(m, cls)]


def first_divergence(reference, optimized, z):
    """Where the optimized decoder first departs from the eager one, and which fast path is responsible.

    Both models keep the original module names, so hooks on the residual /
    up blocks and the output conv line up one to one (the wrapped norms are
    skipped: the optimized one includes the SiLU). Then the decode is repeated
    with one fast-path family disabled at a time on a fresh optimized copy,
    so a mismatch names its family without an interactive session.
    """
    watched = (
        "QwenImage21ResidualBlock",
        "QwenImage21ResidualUpBlock",
        "QwenImage21MidBlock",
    )
    outputs = {"reference": [], "optimized": []}
    handles = []
    for tag, model in (("reference", reference), ("optimized", optimized)):
        for name, m in model.named_modules():
            if type(m).__name__ in watched or name == "decoder.conv_out":
                handles.append(
                    m.register_forward_hook(
                        lambda mod, inp, out, tag=tag, name=name: outputs[tag].append(
                            (name, out.detach().clone())
                        )
                    )
                )
    try:
        ref_out, opt_out = reference.decode(z), optimized.decode(z)
    finally:
        for h in handles:
            h.remove()
    blocks = []
    for (name, ref), (_, opt) in zip(
        outputs["reference"], outputs["optimized"], strict=True
    ):
        if ref.shape != opt.shape or not torch.equal(ref, opt):
            diff = (ref.float() - opt.float()).abs()
            blocks.append(
                f"{name}: max|diff| {diff.max().item():.3e}, "
                f"{(diff > 0).sum().item()}/{diff.numel()} elements"
            )
            if len(blocks) == 3:
                break

    def decode_without(family):
        model = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
        saved = {
            name: getattr(vae_opt, name)
            for name in ("_BIAS_RESIDUAL_FUSION", "_UPSAMPLE_BIAS_FUSION")
        }
        try:
            for name in saved:
                gate = vae_opt.BitExactFusionGate(f"isolation {name}")
                if family == name:
                    gate.disable()
                setattr(vae_opt, name, gate)
            for m in model.modules():
                if family == "padding fold" and isinstance(m, vae_opt.FoldedPadConv2d):
                    m._gates["nchw"].disable()
                    m._gates["nhwc"].disable()
                if family == "norm+SiLU tail" and isinstance(
                    m, vae_opt.FusedChannelRMSNormSiLU
                ):
                    m._exact_gate.disable()
            model.decode(z)  # first sight for the remaining families
            return torch.equal(model.decode(z), ref_out)
        finally:
            for name, gate in saved.items():
                setattr(vae_opt, name, gate)

    families = (
        "padding fold",
        "norm+SiLU tail",
        "_BIAS_RESIDUAL_FUSION",
        "_UPSAMPLE_BIAS_FUSION",
    )
    isolation = ", ".join(
        f"without {family}: {'equal' if decode_without(family) else 'differs'}"
        for family in families
    )
    return (
        "decode "
        + ("equal" if torch.equal(ref_out, opt_out) else "differs")
        + f"; eager reproducible on this latent: {torch.equal(reference.decode(z), ref_out)}"
        + "; first divergent blocks: "
        + ("; ".join(blocks) if blocks else "none")
        + "; "
        + isolation
    )


@pytest.fixture(autouse=True)
def deterministic_cudnn():
    # Bit-exactness is asserted against the eager chain, so both sides must run
    # deterministic conv algorithms; cuDNN may otherwise pick split-K engines
    # whose atomics reorder the sum between two calls.
    previous = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = previous


@pytest.fixture
def module_gates(monkeypatch):
    """Fresh copies of the module-level first-sight gates, so tests do not share state."""
    gates = {
        "_BIAS_RESIDUAL_FUSION": vae_opt.BitExactFusionGate("test bias + residual"),
        "_UPSAMPLE_BIAS_FUSION": vae_opt.BitExactFusionGate("test upsampler bias"),
    }
    for name, gate in gates.items():
        monkeypatch.setattr(vae_opt, name, gate)
    return gates


@torch.no_grad()
def test_lossless_decode_is_bit_identical(module_gates):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    # wrappers keep parameter names and values
    assert set(optimized.state_dict()) == set(reference.state_dict())
    for key, value in optimized.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])

    z = latent()
    expected = reference.decode(z)
    assert torch.equal(reference.decode(z), expected), (
        "the eager decode itself is not reproducible on this platform"
    )
    assert torch.equal(optimized.decode(z), expected)

    # Kernels that are exact by construction must have engaged.
    norm_gates = [
        m._exact_gate for m in modules_of(optimized, vae_opt.FusedChannelRMSNormSiLU)
    ]
    assert len(norm_gates) >= 8
    for gate in norm_gates + list(module_gates.values()):
        assert gate.verified and not gate.disabled, gate.name
    # The padding fold is kept only where cuDNN's kernel for the folded
    # descriptor reproduces the padded conv bit for bit (true on SM120, not on
    # H100), so a declined fold is a valid outcome; each conv must have decided.
    fold_gates = [
        m._gates["nchw"] for m in modules_of(optimized, vae_opt.FoldedPadConv2d)
    ]
    assert len(fold_gates) >= 8
    assert all(gate.verified or gate.disabled for gate in fold_gates)

    z.normal_()
    assert torch.equal(optimized.decode(z), reference.decode(z)), first_divergence(
        reference, optimized, z
    )


@torch.no_grad()
def test_mismatch_falls_back_to_the_eager_chain(module_gates, monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    monkeypatch.setattr(
        vae_opt,
        "channel_rmsnorm_finish_silu",
        lambda x, norm, gamma, scale: torch.zeros_like(x),
    )
    monkeypatch.setattr(
        vae_opt, "bias_residual_add", lambda y, bias, h: torch.zeros_like(y)
    )
    z = latent()
    assert torch.equal(optimized.decode(z), reference.decode(z))
    norm_gates = [
        m._exact_gate for m in modules_of(optimized, vae_opt.FusedChannelRMSNormSiLU)
    ]
    assert norm_gates and all(g.disabled and not g.verified for g in norm_gates)
    bias_gate = module_gates["_BIAS_RESIDUAL_FUSION"]
    assert bias_gate.disabled and not bias_gate.verified


@torch.no_grad()
def test_extra_high_decode_is_close_and_lossless_is_restored(module_gates):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    upsamplers = [
        m
        for m in modules_of(optimized, QwenImage21Resample)
        if m.mode.startswith("upsample")
    ]
    assert len(upsamplers) == 4
    assert all(isinstance(m.resample, vae_opt.FusedUpsample2xConv) for m in upsamplers)

    z = latent()
    expected = reference.decode(z)
    with use_vae_fast_path(optimized, True):
        fast = optimized.decode(z)
        assert optimized.decoder._sgl_channels_last
    torch.testing.assert_close(fast.float(), expected.float(), atol=0.05, rtol=0)
    # gate off: layout and kernels revert, output is bit-identical again
    assert torch.equal(optimized.decode(z), expected)
    assert not optimized.decoder._sgl_channels_last


@torch.no_grad()
def test_folded_upsampler_follows_the_weights_and_moves_with_the_module(module_gates):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    z = latent()
    with use_vae_fast_path(optimized, True):
        before = optimized.decode(z)  # builds the folded kernels
    for model in (reference, optimized):
        for m in modules_of(model, QwenImage21Resample):
            if m.mode.startswith("upsample"):
                m.resample[1].weight.zero_()
                m.resample[1].bias.zero_()
    expected = reference.decode(z)
    assert not torch.equal(expected, before)
    with use_vae_fast_path(optimized, True):
        fast = optimized.decode(z)
    torch.testing.assert_close(fast.float(), expected.float(), atol=0.05, rtol=0)

    # CPU offload must not leave the folded kernels, or anything else, on the GPU.
    optimized.to("cpu")
    for m in optimized.modules():
        for value in list(vars(m).values()) + list(m._buffers.values()):
            assert not (isinstance(value, torch.Tensor) and value.is_cuda), type(m)
    optimized.to("cuda")
    with use_vae_fast_path(optimized, True):
        torch.testing.assert_close(
            optimized.decode(z).float(), expected.float(), atol=0.05, rtol=0
        )


def test_fold_rejects_convs_it_cannot_express():
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


def test_other_vaes_pass_through():
    assert vae_opt.maybe_optimize_qwen_image21_vae(torch.nn.Linear(2, 2)) is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
