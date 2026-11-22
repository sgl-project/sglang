"""Check the signed cache record against independently decoded BF16 inputs on replay."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=40, suite="extra-b-test-1-npu-a3")


class TestDSV41Int8Encode(CustomTestCase):
    def test_replay_selects_scales_for_the_current_input(self):
        from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import (
            decode_int8,
            encode_int8,
            metadata_bytes,
        )

        for width, group in ((128, 128), (448, 224)):
            with self.subTest(width=width):
                # Main KV supplies a non-contiguous slice that omits its rotary tail.
                live = torch.zeros(4, width + 64, dtype=torch.bfloat16, device="npu")[
                    :, :width
                ]
                encode_int8(live, group)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    codes, metadata = encode_int8(live, group)
                self.assertEqual(codes.dtype, torch.int8)
                self.assertEqual(metadata.dtype, torch.uint8)
                self.assertEqual(metadata.shape, (4, metadata_bytes(width, group)))

                for seed in (111, 613, 991):
                    gen = torch.Generator().manual_seed(seed)
                    data = torch.randn(4, width + 64, generator=gen)[
                        :, :width
                    ].bfloat16()
                    data[0].zero_()
                    data[1, ::8] *= 2.0**-22
                    data[2] *= 2.0**-35
                    data[3] *= 2.0**26
                    live.copy_(data)
                    graph.replay()
                    torch.npu.synchronize()

                    actual = decode_int8(codes.cpu(), metadata.cpu(), group)
                    ref_codes, ref_meta = encode_int8(data, group)
                    reference = decode_int8(ref_codes, ref_meta, group)
                    # Computing this per row prevents large activations from hiding
                    # a stale palette or an incorrect selector on a small row.
                    error = torch.linalg.vector_norm(
                        actual.float() - data.float(), dim=1
                    )
                    limit = torch.linalg.vector_norm(
                        reference.float() - data.float(), dim=1
                    )
                    self.assertTrue(
                        torch.all(error <= limit * 1.05), (seed, error, limit)
                    )


if __name__ == "__main__":
    unittest.main()
