"""The DeepSeek-V4.1 ViT fusions (RMSNorm, residual adds, SiLU*up) keep the
eager output bits. Attention is a fixed linear stand-in: it is not changed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models import deepseek_v41_vit as vit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="the fused ViT paths require CUDA",
)

ARGS = SimpleNamespace(
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(ARGS.vision_dim, ARGS.vision_dim, bias=False)

    def forward(self, x, cos, sin, metadata):
        return self.proj(x)


def _make_vit(n_layers: int) -> vit.ViT:
    model = vit.ViT.__new__(vit.ViT)
    nn.Module.__init__(model)
    model.rope_dim = ARGS.vision_dim // ARGS.vision_n_heads // 2
    model.rope_theta = ARGS.vision_rope_theta
    model.patch_embed = vit.PatchEmbed(ARGS)
    blocks = []
    for _ in range(n_layers):
        block = vit.Block.__new__(vit.Block)
        nn.Module.__init__(block)
        block.norm1 = vit._rms_norm(ARGS.vision_dim)
        block.attn = _Attention()
        block.norm2 = vit._rms_norm(ARGS.vision_dim)
        block.mlp = vit.MLP(ARGS)
        blocks.append(block)
    model.blocks = nn.ModuleList(blocks)
    model.norm = vit._rms_norm(ARGS.vision_dim)
    return model


def _eager_forward(model: vit.ViT, patches: torch.Tensor) -> torch.Tensor:
    """The ViT forward before the fusions: native norms, eager adds and SiLU."""
    x = model.patch_embed(patches)
    for block in model.blocks:
        x = x + block.attn(block.norm1.forward_native(x), None, None, None)
        gate, up = block.mlp.w1(block.norm2.forward_native(x)).chunk(2, dim=-1)
        x = x + block.mlp.w2(F.silu(gate) * up)
    return model.norm.forward_native(x)


@torch.no_grad()
@pytest.mark.parametrize("n_h, n_w", [(39, 39), (12, 30)])
def test_fused_vit_forward_matches_eager_bits(n_h, n_w):
    torch.manual_seed(0)
    model = _make_vit(n_layers=3).cuda().to(torch.bfloat16)
    for norm in [model.norm, *(b.norm1 for b in model.blocks)]:
        norm.weight.data = torch.randn_like(norm.weight, dtype=torch.float32)
    for block in model.blocks:
        block.norm2.weight.data = torch.randn(
            ARGS.vision_dim, device="cuda", dtype=torch.float32
        )
    patches = torch.randn(n_h * n_w, 3, 14, 14, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(model(patches, n_h, n_w), _eager_forward(model, patches))
