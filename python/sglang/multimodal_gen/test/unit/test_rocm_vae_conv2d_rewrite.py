"""The ROCm Conv3d -> Conv2d VAE rewrite must reproduce ``F.conv3d`` exactly.

The rewrite folds the temporal axis into channels and runs one ``conv2d`` per
output frame group. It is a pure re-association of the same sums, so the only
way it can go wrong is a bookkeeping slip: wrong fold order, the bias applied
twice, a causal cache stitched at the wrong offset, or ``skip_bias`` /
``residual`` contracts silently dropped. Each case below pins one of those.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform


def _make_weight(c_in, c_out, kt=3, kh=3, kw=3, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(c_out, c_in, kt, kh, kw, dtype=torch.float32)
    weight_2d = (
        weight.permute(0, 2, 1, 3, 4)
        .reshape(c_out, kt * c_in, kh, kw)
        .contiguous()
    )
    return weight, weight_2d


def test_batched_conv2d_matches_conv3d_without_padding():
    x = torch.randn(1, 4, 5, 6, 7, dtype=torch.float32)
    weight, weight_2d = _make_weight(4, 6, seed=1)

    expected = F.conv3d(x, weight, None, stride=(1, 1, 1))
    got = RocmPlatform._conv3d_as_batched_conv2d(
        x, weight_2d, None, (1, 1, 1), kt=3
    )

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_batched_conv2d_matches_conv3d_with_bias_and_temporal_stride():
    x = torch.randn(2, 3, 6, 5, 4, dtype=torch.float32)
    weight, weight_2d = _make_weight(3, 5, seed=2)
    bias = torch.randn(5, dtype=torch.float32)

    expected = F.conv3d(x, weight, bias, stride=(2, 1, 1))
    got = RocmPlatform._conv3d_as_batched_conv2d(
        x, weight_2d, bias, (2, 1, 1), kt=3
    )

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


class _CausalConv3d(nn.Conv3d):
    """Minimal stand-in for ``WanCausalConv3d``: pads the front of time only."""

    def __init__(self, c_in, c_out, kt=3, kh=3, kw=3):
        super().__init__(c_in, c_out, (kt, kh, kw), padding=0)
        # Same layout as the Wan wrapper: (W-, W+, H-, H+, T-, 0).
        self._padding = (kw // 2, kw // 2, kh // 2, kh // 2, 2 * (kt // 2), 0)

    def forward(self, x, cache_x=None, *, residual=None, skip_bias=False):
        raise AssertionError("the rewrite must install its own forward")


def _reference(x, conv, residual=None, skip_bias=False):
    padded = F.pad(x, list(conv._padding))
    out = F.conv3d(
        padded, conv.weight, None if skip_bias else conv.bias, stride=conv.stride
    )
    return out if residual is None else out + residual


def test_rewrite_matches_conv3d_and_keeps_the_wrapper_contract():
    torch.manual_seed(3)
    conv = _CausalConv3d(4, 5)
    with torch.no_grad():
        conv.bias.copy_(torch.randn(5))
    model = nn.Sequential(conv)

    patched = RocmPlatform._replace_conv3d_with_conv2d(model)
    assert patched == 1

    x = torch.randn(1, 4, 4, 6, 7, dtype=torch.float32)
    residual = torch.randn(1, 5, 4, 6, 7, dtype=torch.float32)

    for kwargs in ({}, {"skip_bias": True}, {"residual": residual}):
        expected = _reference(x, conv, **kwargs)
        got = model[0](x, **kwargs)
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), kwargs


def test_rewrite_stitches_the_causal_cache_at_the_right_offset():
    torch.manual_seed(4)
    conv = _CausalConv3d(3, 4)
    with torch.no_grad():
        conv.bias.copy_(torch.randn(4))
    model = nn.Sequential(conv)
    assert RocmPlatform._replace_conv3d_with_conv2d(model) == 1

    x = torch.randn(1, 3, 4, 5, 6, dtype=torch.float32)
    cache_x = torch.randn(1, 3, 2, 5, 6, dtype=torch.float32)

    # The cached frames *are* the causal history, so the front time pad shrinks
    # by exactly as many frames as the cache supplies.
    pad = list(conv._padding)
    pad[4] -= cache_x.shape[2]
    expected = F.conv3d(
        F.pad(torch.cat([cache_x, x], dim=2), pad), conv.weight, conv.bias
    )
    got = model[0](x, cache_x=cache_x)

    assert got.shape == expected.shape == (1, 4, 4, 5, 6)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_rewrite_leaves_non_3d_kernels_and_grouped_convs_alone():
    pointwise = _CausalConv3d(3, 4, kt=1, kh=1, kw=1)
    grouped = _CausalConv3d(4, 4)
    grouped.groups = 2
    model = nn.Sequential(pointwise, grouped)

    assert RocmPlatform._replace_conv3d_with_conv2d(model) == 0


if __name__ == "__main__":
    test_batched_conv2d_matches_conv3d_without_padding()
    test_batched_conv2d_matches_conv3d_with_bias_and_temporal_stride()
    test_rewrite_matches_conv3d_and_keeps_the_wrapper_contract()
    test_rewrite_stitches_the_causal_cache_at_the_right_offset()
    test_rewrite_leaves_non_3d_kernels_and_grouped_convs_alone()
    print("ok")
