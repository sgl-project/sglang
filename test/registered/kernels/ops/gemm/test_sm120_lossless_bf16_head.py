"""Lossless BF16 bit patterns, original arithmetic, and stable graph buffers."""

import unittest

import numpy as np
import torch

from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import (
    _module,
    _refresh_lossless_buffers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestLosslessBf16Head(CustomTestCase):
    def test_every_bf16_bit_pattern_and_raw_fallback(self):
        bits = np.tile(np.arange(65536, dtype=np.uint16), 4)
        # A block spanning both exponent extremes must use raw 16-bit storage.
        bits[:256:2] = 0x0001
        bits[1:256:2] = 0xFFBF
        weight = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            weight.reshape(64, 4096).cuda(), requires_grad=False
        )
        original = layer.weight.view(torch.int16).clone()
        _refresh_lossless_buffers(layer)
        restored = torch.empty_like(layer.weight)
        _module(64, 4096).restore(
            layer._lossless_bf16_payload, layer._lossless_bf16_metadata, restored
        )
        torch.testing.assert_close(restored.view(torch.int16), original, rtol=0, atol=0)
        torch.testing.assert_close(
            layer.weight.view(torch.int16), original, rtol=0, atol=0
        )
        metadata = layer._lossless_bf16_metadata.cpu().numpy()
        compressed = (metadata & (1 << 40)) != 0
        self.assertTrue(compressed.any())
        self.assertTrue((~compressed).any())
        self.assertEqual(list(layer.state_dict()), ["weight"])

    def test_graph_repack_keeps_pointers_and_original_gemv_results(self):
        torch.manual_seed(61)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        _refresh_lossless_buffers(layer)
        pointers = (
            layer._lossless_bf16_payload.data_ptr(),
            layer._lossless_bf16_metadata.data_ptr(),
        )
        module = _module(64, 4096)
        x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((1, 64), device="cuda", dtype=torch.bfloat16)

        def run():
            module.run(
                x, layer._lossless_bf16_payload, layer._lossless_bf16_metadata, out
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for case in range(4):
            x.normal_()
            if case == 0:
                x.zero_()
            with torch.no_grad():
                layer.weight.normal_()
                if case % 2:
                    # Force many blocks into raw storage, changing all offsets.
                    layer.weight.view(torch.int16)[:, ::256] = 1
            _refresh_lossless_buffers(layer)
            self.assertEqual(
                pointers,
                (
                    layer._lossless_bf16_payload.data_ptr(),
                    layer._lossless_bf16_metadata.data_ptr(),
                ),
            )
            expected = hopper_bf16_gemv(x, layer.weight)
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            run()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
