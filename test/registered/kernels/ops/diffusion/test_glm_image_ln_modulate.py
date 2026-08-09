"""GLM-Image fused LN+modulate / qk-LN fast paths must stay bit-exact vs eager."""

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.glm_image as glm_image
from sglang.multimodal_gen.runtime.models.dits.glm_image import (
    _eager_ln_modulate,
    _glm_ln_modulate,
    _glm_qk_layernorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(1, 4096, 4096), (2, 301, 4096), (1, 1, 2560)])
def test_fused_ln_modulate_is_bit_exact(shape):
    # (1, 4096, 4096) is the real GLM-Image image-stream shape (1024^2,
    # hidden 4096); the others cover the text stream and another hidden.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 8).bfloat16()
    emb = torch.randn(batch, 12 * hidden, device="cuda").bfloat16()
    chunks = emb.chunk(12, dim=1)  # strided adaLN projection views
    shift, scale = chunks[0], chunks[2]
    out = _glm_ln_modulate(norm, x, scale, shift, x.dtype)
    assert torch.equal(out, _eager_ln_modulate(norm, x, scale, shift, x.dtype))
    assert glm_image._GLM_LN_MOD.verified
    assert not glm_image._GLM_LN_MOD.disabled


@pytest.mark.parametrize("shape", [(1, 4360, 32, 128), (2, 37, 3, 40), (1, 129, 5, 64)])
def test_fused_qk_head_layernorm_is_bit_exact(shape):
    # (1, 4360, 32, 128) is the real GLM-Image q/k shape (text + image
    # tokens, 32 heads of dim 128); the others cover partially-filled warps.
    torch.manual_seed(1)
    batch, seq, heads, head_dim = shape
    norm_q = torch.nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False).cuda()
    norm_k = torch.nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False).cuda()
    q = (torch.randn(batch, seq, heads, head_dim, device="cuda") * 5).bfloat16()
    k = (torch.randn(batch, seq, heads, head_dim, device="cuda") * 5).bfloat16()
    q_out, k_out = _glm_qk_layernorm(norm_q, norm_k, q, k, q.dtype)
    assert torch.equal(q_out, norm_q(q).to(q.dtype))
    assert torch.equal(k_out, norm_k(k).to(k.dtype))
    assert glm_image._GLM_QK_LN.verified
    assert not glm_image._GLM_QK_LN.disabled


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
