# SPDX-License-Identifier: Apache-2.0
"""The Qwen-Image 2.1 VAE decoder fast paths must leave the decode bit-identical.

Regressions caught: a fused norm + SiLU tail or a folded-padding conv that no
longer matches the original chain, parameter names changed by the wrappers,
and a fast path that stays on after its first-sight check failed.
"""
from copy import deepcopy

import pytest
import torch

from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import qwen_image21_vae_cuda_opt as vae_opt
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
)
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
    return [m._gate for m in vae.modules() if isinstance(m, (vae_opt.FusedChannelRMSNormSiLU, vae_opt.FoldedPadConv2d))]


@torch.no_grad()
def test_decode_is_bit_identical_and_paths_verify():
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
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
    z.normal_()
    assert torch.equal(optimized.decode(z), reference.decode(z))


@torch.no_grad()
def test_mismatch_disables_the_norm_fast_path(monkeypatch):
    reference = make_vae()
    optimized = vae_opt.maybe_optimize_qwen_image21_vae(deepcopy(reference))
    monkeypatch.setattr(
        vae_opt, "channel_rmsnorm_finish_silu", lambda x, norm, gamma, scale: torch.zeros_like(x)
    )
    z = torch.randn(1, 4, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(optimized.decode(z), reference.decode(z))
    norm_gates = [m._gate for m in optimized.modules() if isinstance(m, vae_opt.FusedChannelRMSNormSiLU)]
    assert norm_gates and all(gate.disabled and not gate.verified for gate in norm_gates)


@torch.no_grad()
def test_other_vaes_pass_through():
    assert vae_opt.maybe_optimize_qwen_image21_vae(torch.nn.Linear(2, 2)) is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
