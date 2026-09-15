import unittest

import torch

from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import LTX2VideoCausalConv3d


@unittest.skipUnless(
    hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
)
class TestLTX2CausalConvChannelsLast(unittest.TestCase):
    """The channels_last_3d causal-conv path must stay numerically identical to
    the original repeat()+concatenate() temporal padding (only the conv kernel's
    floating-point accumulation order may differ, which fp32 keeps negligible)."""

    def _check(self, causal: bool, kernel_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        conv = LTX2VideoCausalConv3d(8, 8, kernel_size).to(device, torch.float32).eval()
        x = torch.randn(1, 8, 5, 6, 7, dtype=torch.float32, device=device)

        # Default (contiguous) weight -> original repeat/concat padding branch.
        self.assertFalse(conv._weight_is_channels_last_3d())
        with torch.no_grad():
            y_ref = conv(x.clone(), causal=causal)

        # channels_last_3d weight -> layout-preserving pad branch.
        conv.conv.weight.data = conv.conv.weight.data.to(
            memory_format=torch.channels_last_3d
        )
        self.assertTrue(conv._weight_is_channels_last_3d())
        with torch.no_grad():
            y_cl = conv(x.clone(), causal=causal)

        self.assertEqual(y_ref.shape, y_cl.shape)
        # On CUDA, cuDNN preserves channels_last_3d through the conv; CPU conv3d
        # returns a contiguous tensor regardless. The pad output layout itself is
        # asserted device-independently in test_pad_replicates_edge_frames_exactly.
        if device == "cuda":
            self.assertTrue(y_cl.is_contiguous(memory_format=torch.channels_last_3d))
        # Equal weights in either memory format -> equal math; only the conv
        # kernel's float accumulation order may differ (negligible in fp32).
        torch.testing.assert_close(y_ref, y_cl, rtol=1e-4, atol=1e-4)

    def test_causal_matches_reference(self):
        self._check(causal=True, kernel_size=3)

    def test_non_causal_matches_reference(self):
        self._check(causal=False, kernel_size=3)

    def test_temporal_only_kernel_matches_reference(self):
        self._check(causal=True, kernel_size=(3, 1, 1))

    def test_causal_cache_matches_monolithic_reference(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        conv = LTX2VideoCausalConv3d(4, 5, 3).to(device, torch.float32).eval()
        conv.conv.weight.data = conv.conv.weight.data.to(
            memory_format=torch.channels_last_3d
        )
        self.assertTrue(conv._weight_is_channels_last_3d())
        x = torch.randn(1, 4, 6, 3, 3, dtype=torch.float32, device=device)

        with torch.no_grad():
            whole = conv(x, causal=True)
            cache = {}
            parts = [
                conv(chunk, causal=True, conv_cache=cache, cache_key="conv")
                for chunk in (x[:, :, :2], x[:, :, 2:])
            ]
            chunked = torch.cat(parts, dim=2)

        self.assertEqual(chunked.shape, whole.shape)
        torch.testing.assert_close(whole, chunked, rtol=1e-4, atol=1e-4)

    def test_pad_replicates_edge_frames_exactly(self):
        # The temporal pad must replicate the first (and, when non-causal, last)
        # frame exactly -- assert against an explicit reference construction.
        conv = LTX2VideoCausalConv3d(4, 4, 3).to(torch.float32)
        conv.conv.weight.data = conv.conv.weight.data.to(
            memory_format=torch.channels_last_3d
        )
        x = torch.randn(1, 4, 3, 2, 2, dtype=torch.float32)
        padded = conv._causal_temporal_pad_channels_last(x, left=2, right=1)
        expected = torch.cat(
            [x[:, :, :1].repeat(1, 1, 2, 1, 1), x, x[:, :, -1:]], dim=2
        )
        self.assertTrue(padded.is_contiguous(memory_format=torch.channels_last_3d))
        torch.testing.assert_close(padded, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
