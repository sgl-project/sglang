# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits import joy_image


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("batch,tokens", [(1, 8048), (2, 4096)])
@torch.inference_mode()
def test_strided_qk_preserves_inputs_and_replays_updated_parameters(
    batch, tokens, monkeypatch
):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    torch.manual_seed(831)
    gate = BitExactFusionGate("test Joy QK", per_signature=True)
    monkeypatch.setattr(joy_image, "_JOY_IMAGE_QK_ROPE", gate)
    qkv = torch.randn(batch, tokens, 3, 32, 128, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.unbind(2)
    qnorm = joy_image.RMSNorm(128, eps=1e-6).to(device="cuda", dtype=q.dtype)
    knorm = joy_image.RMSNorm(128, eps=1e-6).to(device="cuda", dtype=q.dtype)
    qnorm.weight.copy_(torch.randn_like(qnorm.weight))
    knorm.weight.copy_(torch.randn_like(knorm.weight))
    angles = torch.randn(tokens, 64, device="cuda", dtype=torch.float32)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    def reference():
        return joy_image.apply_qk_norm_with_optional_rope(
            q=q.contiguous(),
            k=k.contiguous(),
            q_norm=qnorm,
            k_norm=knorm,
            head_dim=128,
            cos_sin_cache=cache,
            freqs_complex=None,
            is_neox=False,
            allow_inplace=True,
        )

    pristine = qkv.clone()
    actual = joy_image._joy_image_qk_rope(q, k, qnorm, knorm, cache, None)
    expected = reference()
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.view(torch.int16), right.view(torch.int16))
    assert torch.equal(qkv.view(torch.int16), pristine.view(torch.int16))
    assert gate.verified_sigs and not gate.disabled

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = joy_image._joy_image_qk_rope(q, k, qnorm, knorm, cache, None)
    qkv.add_(0.125)
    qnorm.weight.mul_(1.125)
    knorm.weight.add_(0.25)
    cache.neg_()
    pristine.copy_(qkv)
    graph.replay()
    expected = reference()
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.view(torch.int16), right.view(torch.int16))
    assert torch.equal(qkv.view(torch.int16), pristine.view(torch.int16))
