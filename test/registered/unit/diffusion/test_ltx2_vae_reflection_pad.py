# SPDX-License-Identifier: Apache-2.0
"""Keep LTX spatial reflection padding exact without NCDHW layout round trips."""

import unittest

import torch

from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import LTX2VideoCausalConv3d
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestLTX2VAEReflectionPad(CustomTestCase):
    def test_convolution_matches_native_padding(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for causal in (False, True):
                for channels_last in (False, True):
                    with self.subTest(
                        dtype=dtype, causal=causal, channels_last=channels_last
                    ):
                        conv = (
                            LTX2VideoCausalConv3d(
                                8,
                                12,
                                kernel_size=(3, 3, 5),
                                spatial_padding_mode="reflect",
                            )
                            .cuda()
                            .to(dtype)
                        )
                        if channels_last:
                            conv.to(memory_format=torch.channels_last_3d)
                        x = torch.randn(2, 8, 5, 11, 15, device="cuda", dtype=dtype)
                        left, right = (2, 0) if causal else (1, 1)
                        if channels_last:
                            padded = conv._causal_temporal_pad_channels_last(
                                x, left, right
                            )
                        else:
                            padded = torch.cat(
                                [
                                    x[:, :, :1].expand(-1, -1, left, -1, -1),
                                    x,
                                    x[:, :, -1:].expand(-1, -1, right, -1, -1),
                                ],
                                dim=2,
                            )
                        expected = conv.conv(padded)
                        actual = conv(x, causal=causal)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streamed_chunks_preserve_cache_and_output(self):
        conv = (
            LTX2VideoCausalConv3d(8, 8, spatial_padding_mode="reflect")
            .cuda()
            .to(dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
        )
        chunks = [
            torch.randn(1, 8, t, 7, 9, device="cuda", dtype=torch.bfloat16)
            for t in (1, 3, 1, 2)
        ]
        cache = {}
        previous = None
        for x in chunks:
            padded = conv._causal_temporal_pad_channels_last(x, 2, 0, left_pad=previous)
            expected = conv.conv(padded)
            actual = conv(x, conv_cache=cache, cache_key="conv")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            previous = padded[:, :, -2:].detach().clone()
            torch.testing.assert_close(cache["conv"], previous, rtol=0, atol=0)

    def test_cuda_graph_replay_uses_changed_frames(self):
        conv = (
            LTX2VideoCausalConv3d(8, 8, spatial_padding_mode="reflect")
            .cuda()
            .to(dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
        )
        x = torch.randn(1, 8, 5, 7, 9, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                conv(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = conv(x)
        x.normal_()
        expected = conv.conv(conv._causal_temporal_pad_channels_last(x, 2, 0))
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
