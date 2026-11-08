from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention import vision
from sglang.srt.models import deepseek_v41_vit
from sglang.srt.models.deepseek_v41_vit import Aligner, ViT, encode_images
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VarlenSdpa(torch.nn.Module):
    """CPU stand-in for the varlen GPU kernels: attention stays inside each
    forward_metadata.cu_seqlens block."""

    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, q, k, v, *, forward_metadata, softmax_scale=None, **_kwargs):
        # [tokens, heads, head_dim] -> [heads, tokens, head_dim] per sequence.
        bounds = forward_metadata.cu_seqlens.tolist()
        outputs = [
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
                scale=softmax_scale,
            ).transpose(0, 1)
            for start, end in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(outputs)


def test_packed_images_match_per_image_encoding(monkeypatch, request):
    """A packed pass must keep each image's attention, RoPE positions and 2x2
    merge to itself, across pass boundaries, including grids that need padding."""
    override = get_context().override_server_args(mm_attention_backend="sdpa")
    override.install()
    request.addfinalizer(override.restore)
    monkeypatch.setitem(vision.QKV_BACKEND_IMPL, "sdpa", _VarlenSdpa)
    monkeypatch.setattr(
        deepseek_v41_vit, "_PACKED_ATTENTION_BACKENDS", frozenset({"sdpa"})
    )
    monkeypatch.setattr(deepseek_v41_vit, "_MAX_IMAGES_PER_PASS", 2)

    args = SimpleNamespace(
        vision_dim=32,
        vision_n_heads=2,
        vision_patch_size=2,
        vision_inter_dim=16,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        dim=24,
    )
    torch.manual_seed(0)
    grids = [(3, 5), (4, 4), (1, 2)]
    patches = [torch.randn(h * w, 3, 2, 2) for h, w in grids]

    with (
        torch.no_grad(),
        get_parallel().override(
            tp_size=1, tp_rank=0, tp_group=None, attn_tp_size=1, attn_tp_rank=0
        ),
    ):
        vit, aligner = ViT(args).float(), Aligner(args).float()
        # Parallel linears allocate with torch.empty.
        for param in [*vit.parameters(), *aligner.parameters()]:
            param.normal_(std=0.1)
        packed = encode_images(
            vision=vit, aligner=aligner, patches=patches, grids=grids
        )
        per_image = [aligner(vit(p, h, w), h, w) for p, (h, w) in zip(patches, grids)]

    assert [t.shape for t in packed] == [t.shape for t in per_image]
    for got, want in zip(packed, per_image):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
