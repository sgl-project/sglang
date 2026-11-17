from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch
import torch.nn as nn

import sglang.srt.layers.conv as conv_module
from sglang.srt.layers.conv import Conv2dLayer, Conv3dLayer
from sglang.test.test_utils import CustomTestCase


def _copy_weights(src, dst_nn):
    """Copy weights from Conv*dLayer to nn.Conv*d for comparison."""
    with torch.no_grad():
        dst_nn.weight.copy_(src.weight)
        if src.bias is not None:
            dst_nn.bias.copy_(src.bias)


class TestConv3dRuntimePolicy(CustomTestCase):
    def _fallback_required(
        self,
        *,
        torch_version: tuple[int, ...],
        cudnn_version: int | None,
        is_cuda: bool = True,
        is_hip: bool = False,
    ) -> bool:
        with (
            mock.patch.object(conv_module, "is_cuda", return_value=is_cuda),
            mock.patch.object(conv_module, "is_hip", return_value=is_hip),
            mock.patch.object(conv_module, "torch_release", torch_version),
            mock.patch.object(
                conv_module.torch.backends.cudnn,
                "version",
                return_value=cudnn_version,
            ),
        ):
            return conv_module._is_conv3d_linear_fallback_required()

    def test_torch_2_9_0_requires_fallback(self):
        self.assertTrue(
            self._fallback_required(torch_version=(2, 9, 0), cudnn_version=91500)
        )

    def test_torch_2_8_keeps_conv3d(self):
        self.assertFalse(
            self._fallback_required(torch_version=(2, 8, 0), cudnn_version=91400)
        )

    def test_torch_2_9_1_with_old_cudnn_requires_fallback(self):
        self.assertTrue(
            self._fallback_required(torch_version=(2, 9, 1), cudnn_version=91400)
        )

    def test_torch_2_9_1_with_fixed_cudnn_keeps_conv3d(self):
        self.assertFalse(
            self._fallback_required(torch_version=(2, 9, 1), cudnn_version=91500)
        )

    def test_torch_2_10_keeps_conv3d(self):
        self.assertFalse(
            self._fallback_required(torch_version=(2, 10, 0), cudnn_version=91400)
        )

    def test_non_cuda_runtime_keeps_conv3d(self):
        self.assertFalse(
            self._fallback_required(
                torch_version=(2, 9, 1), cudnn_version=91400, is_cuda=False
            )
        )

    def test_hip_runtime_keeps_conv3d(self):
        self.assertFalse(
            self._fallback_required(
                torch_version=(2, 9, 1),
                cudnn_version=91400,
                is_cuda=True,
                is_hip=True,
            )
        )

    def test_unknown_cudnn_uses_performance_safe_fallback(self):
        self.assertTrue(
            self._fallback_required(torch_version=(2, 9, 1), cudnn_version=None)
        )

    def _make_layer(self, *, stride=(2, 2, 2)) -> Conv3dLayer:
        with (
            mock.patch.object(
                conv_module, "_is_conv3d_linear_fallback_required", return_value=True
            ),
            mock.patch.object(conv_module, "print_warning_once") as warning,
        ):
            layer = Conv3dLayer(
                3, 8, kernel_size=(2, 2, 2), stride=stride, disable_linear=True
            )
        self._init_warning = warning
        return layer

    def _picks_linear(self, layer) -> dict[str, bool]:
        """Run both dispatch targets and report which branch each one took."""
        x = object()
        picks = {}
        for name in ("forward_native", "forward_cuda"):
            with (
                mock.patch.object(layer, "_forward_mulmat") as linear,
                mock.patch.object(layer, "_forward_conv") as conv,
                mock.patch.object(conv_module, "print_warning_once") as warning,
            ):
                getattr(layer, name)(x)
            self.assertNotEqual(linear.called, conv.called)
            warning.assert_not_called()
            picks[name] = linear.called
        return picks

    def test_override_warns_once_at_construction(self):
        self._make_layer()
        self._init_warning.assert_called_once()

    def test_native_and_cuda_paths_both_use_linear_fallback(self):
        """torch.compile runs forward_native, so it must match forward_cuda."""
        layer = self._make_layer()
        self.assertEqual(
            self._picks_linear(layer),
            {"forward_native": True, "forward_cuda": True},
        )

    def test_non_equivalent_conv_cannot_use_linear_fallback(self):
        layer = self._make_layer(stride=1)
        self._init_warning.assert_not_called()
        self.assertEqual(
            self._picks_linear(layer),
            {"forward_native": False, "forward_cuda": False},
        )


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
