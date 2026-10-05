"""Qwen3-MoE's XPU fused QK-norm + RoPE call against the model's unfused path.

Kernel correctness is tested in sgl-kernel-xpu; this guards the sglang side of
the contract: ``RotaryEmbedding.cos_sin_cache`` (including RoPE scaling baked
into it) is what ``sgl_kernel.fused_inplace_qknorm_rope`` reads, fed the way
``Qwen3MoeAttention.apply_qk_norm_rope`` feeds it.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="nightly-xpu-1-gpu", nightly=True)

if is_xpu():
    from sgl_kernel import fused_inplace_qknorm_rope

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.models.utils import apply_qk_norm

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"
DTYPE = torch.bfloat16
EPS = 1e-6
# Same tolerance as kernels/ops/diffusion/test_rope.py: the fused kernel skips
# the unfused path's bf16 rounding between norm and RoPE.
ATOL = 8e-2
RTOL = 1e-2
YARN_SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}


@pytest.mark.parametrize("rope_scaling", [None, YARN_SCALING], ids=["default", "yarn"])
# head_dim >= 128: below that, RotaryEmbedding.forward_xpu takes the bf16-cache fallback.
@pytest.mark.parametrize("head_dim,rotary_dim", [(128, 128), (128, 64)])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 257])
def test_qknorm_rope_matches_model_unfused_path(
    rope_scaling,
    head_dim: int,
    rotary_dim: int,
    is_neox: bool,
    num_tokens: int,
) -> None:
    """The fused call fed from a model's RotaryEmbedding matches its unfused apply_qk_norm + rotary_emb path."""
    num_q_heads, num_kv_heads = 8, 1
    max_position = 131072
    rotary_emb = get_rope(
        head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=1e6,
        is_neox_style=is_neox,
        rope_scaling=rope_scaling,
    ).to(DEVICE)
    q_norm = RMSNorm(head_dim, eps=EPS).to(DEVICE).to(DTYPE)
    k_norm = RMSNorm(head_dim, eps=EPS).to(DEVICE).to(DTYPE)
    q_norm.weight.data.normal_(mean=1.0, std=0.3)
    k_norm.weight.data.normal_(mean=1.0, std=0.3)

    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE)

    q_ref, k_ref, _ = qkv.clone().split([q_size, kv_size, kv_size], dim=-1)
    q_ref, k_ref = apply_qk_norm(
        q=q_ref, k=k_ref, q_norm=q_norm, k_norm=k_norm, head_dim=head_dim
    )
    q_ref, k_ref = rotary_emb(positions, q_ref, k_ref)

    fused = qkv.clone()
    q, k, v = fused.split([q_size, kv_size, kv_size], dim=-1)
    fused_inplace_qknorm_rope(
        q=q.view(-1, num_q_heads, head_dim),
        k=k.view(-1, num_kv_heads, head_dim),
        q_weight=q_norm.weight,
        k_weight=k_norm.weight,
        cos_sin_cache=rotary_emb.cos_sin_cache,
        positions=positions,
        is_neox=rotary_emb.is_neox_style,
        eps=q_norm.variance_epsilon,
        head_dim=head_dim,
        rope_dim=rotary_emb.rotary_dim,
    )

    torch.testing.assert_close(q, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k, k_ref, atol=ATOL, rtol=RTOL)
    assert torch.equal(v, qkv[:, q_size + kv_size :])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
