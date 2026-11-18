from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch
import torch.nn as nn

from sglang.srt.layers.conv import Conv2dLayer, Conv3dLayer


def _copy_weights(src, dst_nn):
    """Copy weights from Conv*dLayer to nn.Conv*d for comparison."""
    with torch.no_grad():
        dst_nn.weight.copy_(src.weight)
        if src.bias is not None:
            dst_nn.bias.copy_(src.bias)


class TestConv2dLayer(unittest.TestCase):
    def test_basic_patch_embedding(self):
        layer = Conv2dLayer(3, 768, kernel_size=14, stride=14, bias=False)
        ref = nn.Conv2d(3, 768, kernel_size=14, stride=14, bias=False)
        self.assertFalse(layer.enable_linear)
        _copy_weights(layer, ref)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x), ref(x), rtol=1e-4, atol=1e-4
            )

    def test_enable_linear(self):
        layer = Conv2dLayer(
            3, 768, kernel_size=14, stride=14, bias=True, disable_linear=False
        )
        ref = nn.Conv2d(3, 768, kernel_size=14, stride=14, bias=True)
        self.assertTrue(layer.enable_linear)
        _copy_weights(layer, ref)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x), ref(x), rtol=1e-4, atol=1e-4
            )

    def test_mulmat_matches_conv(self):
        layer = Conv2dLayer(
            3, 768, kernel_size=14, stride=14, bias=True, disable_linear=False
        )
        self.assertTrue(layer.enable_linear)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(
                layer._forward_mulmat(x),
                layer._forward_conv(x),
                rtol=1e-4,
                atol=1e-4,
            )


class TestConvValidation(unittest.TestCase):
    def test_in_channels_not_divisible_by_groups(self):
        with self.assertRaises(ValueError):
            Conv2dLayer(3, 64, kernel_size=3, stride=1, groups=2)

    def test_out_channels_not_divisible_by_groups(self):
        with self.assertRaises(ValueError):
            Conv2dLayer(4, 6, kernel_size=3, stride=1, groups=4)

    def test_invalid_padding_string(self):
        with self.assertRaises(ValueError):
            Conv2dLayer(3, 64, kernel_size=3, stride=1, padding="full")

    def test_padding_same_with_stride(self):
        with self.assertRaises(ValueError):
            Conv2dLayer(3, 64, kernel_size=3, stride=2, padding="same")

    def test_padding_same_with_non_zeros_padding_mode(self):
        layer = Conv2dLayer(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=True,
        )
        ref = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=True,
        )
        self.assertFalse(layer.enable_linear)
        _copy_weights(layer, ref)
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x), ref(x), rtol=1e-4, atol=1e-4
            )

    def test_invalid_padding_mode(self):
        with self.assertRaises(ValueError):
            Conv3dLayer(3, 64, kernel_size=3, stride=1, padding_mode="invalid")

    def test_conv3d_in_channels_not_divisible_by_groups(self):
        with self.assertRaises(ValueError):
            Conv3dLayer(3, 64, kernel_size=3, stride=1, groups=2)


class TestConv3dLayer(unittest.TestCase):
    def test_basic_temporal_patch_embedding(self):
        layer = Conv3dLayer(
            3, 1152, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=False
        )
        ref = nn.Conv3d(
            3, 1152, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=False
        )
        self.assertTrue(layer.enable_linear)
        _copy_weights(layer, ref)
        x = torch.randn(1, 3, 2, 14, 14)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x), ref(x), rtol=1e-4, atol=1e-4
            )

    def test_with_bias(self):
        layer = Conv3dLayer(
            3, 1536, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=True
        )
        ref = nn.Conv3d(3, 1536, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=True)
        self.assertTrue(layer.enable_linear)
        _copy_weights(layer, ref)
        x = torch.randn(4, 3, 2, 14, 14)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x), ref(x), rtol=1e-4, atol=1e-4
            )

    def test_mulmat_matches_conv(self):
        layer = Conv3dLayer(
            3, 1152, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=True
        )
        self.assertTrue(layer.enable_linear)
        x = torch.randn(2, 3, 2, 14, 14)
        with torch.no_grad():
            torch.testing.assert_close(
                layer._forward_mulmat(x),
                layer._forward_conv(x),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_glm4v_workflow(self):
        """GLM4V-style: 2D input -> reshape to 5D -> Conv3dLayer -> flatten."""
        in_channels, temporal_patch_size, patch_size = 3, 2, 14
        hidden_size = 1536
        layer = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=True,
        )
        ref = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=True,
        )
        _copy_weights(layer, ref)
        num_patches = 4
        flat_dim = in_channels * temporal_patch_size * patch_size * patch_size
        x_2d = torch.randn(num_patches, flat_dim)
        x_5d = x_2d.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
        with torch.no_grad():
            torch.testing.assert_close(
                layer.forward_native(x_5d).view(-1, hidden_size),
                ref(x_5d).view(-1, hidden_size),
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
