"""Non-causal image boundaries and original head scaling on Volta."""

import importlib.util

import pytest
import torch

from sglang.srt.layers.attention.vision import (
    SingletonCache,
    VisionFlashV100Attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or importlib.util.find_spec("flash_attn_v100") is None,
    reason="Requires V100 and the optional FlashAttention library",
)


@pytest.mark.parametrize("head_dim", [64, 72, 128])
@pytest.mark.parametrize("scale", [None, 0.07])
def test_vision_attention_ragged_fp32_reference(head_dim, scale):
    torch.manual_seed(401 + head_dim)
    lengths = [3, 17, 131]
    packed = torch.randn(
        sum(lengths), 3, 4, head_dim, device="cuda", dtype=torch.float16
    )
    q, k, v = packed.unbind(1)
    original = packed.clone()
    cu = torch.tensor([0, 3, 20, 151], dtype=torch.int32, device="cuda")
    output = VisionFlashV100Attention()(
        q, k, v, cu, 1, sum(lengths), softmax_scale=scale
    )
    expected = []
    start = 0
    for length in lengths:
        qq, kk, vv = [
            x[start : start + length].float().transpose(0, 1) for x in (q, k, v)
        ]
        probabilities = (
            qq @ kk.transpose(-1, -2) * (scale if scale is not None else head_dim**-0.5)
        ).softmax(-1)
        expected.append((probabilities @ vv).transpose(0, 1))
        start += length
    expected = torch.cat(expected)
    assert output.shape == q.shape and output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.002)
    assert (output.float() - expected).norm() / expected.norm() < 0.002


@pytest.mark.parametrize("metadata", ["none", "cpu", "cached"])
def test_vision_is_bidirectional_and_isolates_images(metadata):
    q = torch.zeros(10, 4, 72, dtype=torch.float16, device="cuda")
    k = torch.zeros_like(q)
    v = (
        torch.arange(10, dtype=torch.float16, device="cuda")[:, None, None]
        .expand_as(q)
        .contiguous()
    )
    cu = torch.tensor([0, 5, 10], dtype=torch.int32)
    if metadata == "none":
        cu = None
    elif metadata == "cached":
        cu = SingletonCache(cu.cuda())
    output = VisionFlashV100Attention()(q, k, v, cu, 2, 5)
    expected = torch.cat([torch.full_like(q[:5], 2), torch.full_like(q[5:], 7)])
    torch.testing.assert_close(output, expected, rtol=0.001, atol=0.001)
