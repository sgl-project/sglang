"""Protect the local-row offset, padding and fallback contracts on Ascend."""

import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers import vocab_parallel_embedding as embedding
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=240, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU required")
class TestMaskedInputNPU(CustomTestCase):
    def check_case(self, tokens, cfg):
        # Exercise the unchanged eager body, including its dtype promotion,
        # rather than copying the fused kernel's formula as the only oracle.
        with patch.object(embedding, "_is_npu", False):
            expected = embedding.get_masked_input_and_mask(tokens, *cfg)
        actual = embedding.get_masked_input_and_mask(tokens, *cfg)
        for result, reference in zip(actual, expected):
            self.assertEqual(result.shape, tokens.shape)
            self.assertEqual(result.dtype, reference.dtype)
            self.assertEqual(result.device, tokens.device)
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        return actual

    def test_boundaries_and_padding(self):
        # Added rows begin at local 12, after 8 original rows and 4 padding
        # rows. Endpoints are exclusive; padding ids are not valid tokens.
        ids = [9, 10, 17, 18, 21, 99, 100, 102, 103, -1]
        expected_ids = [0, 0, 7, 0, 0, 0, 12, 14, 0, 0]
        expected_mask = [True, False, False, True, True, True, False, False, True, True]
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                tokens = torch.tensor(ids, device="npu", dtype=dtype)
                mapped, mask = self.check_case(tokens, (10, 18, 4, 100, 103))
                self.assertEqual(mapped.dtype, torch.int64)
                self.assertEqual(mapped.cpu().tolist(), expected_ids)
                self.assertEqual(mask.cpu().tolist(), expected_mask)

    def test_shapes_dtypes_and_shards(self):
        # Qwen3.5-35B-A3B has 248320 tokens, exactly 124160 per TP2 rank.
        # Also cover empty/partial shards, nonempty added vocab, and offsets
        # outside signed int32, where narrowing constants silently corrupts ids.
        configs = [
            (0, 124160, 0, 248320, 248320),
            (124160, 248320, 0, 248320, 248320),
            (10, 18, 4, 100, 103),
            (64, 67, 61, 200, 202),
            (128, 128, 64, 200, 200),
            (2**32, 2**32 + 8, 4, 2**33, 2**33 + 3),
        ]
        generator = torch.Generator().manual_seed(20261101)
        for dtype in (torch.int32, torch.int64):
            for cfg in configs:
                if dtype == torch.int32 and cfg[0] >= 2**31:
                    continue
                lo, hi, _, added_lo, added_hi = cfg
                edges = [
                    lo - 1,
                    lo,
                    hi - 1,
                    hi,
                    added_lo - 1,
                    added_lo,
                    added_hi - 1,
                    added_hi,
                ]
                edges += [torch.iinfo(dtype).min, torch.iinfo(dtype).max]
                for shape in [(), (0,), (2, 0, 3), (1,), (32,), (2, 4, 4), (4097,)]:
                    with self.subTest(dtype=dtype, cfg=cfg, shape=shape):
                        cpu = torch.randint(
                            -10, 300000, shape, dtype=dtype, generator=generator
                        )
                        n = min(cpu.numel(), len(edges))
                        cpu.reshape(-1)[:n] = torch.tensor(edges[:n], dtype=dtype)
                        self.check_case(cpu.to("npu"), cfg)

    def test_noncontiguous_and_cpu_fallback(self):
        cfg = (10, 18, 4, 100, 103)
        for device in ("cpu", "npu"):
            for dtype in (torch.int32, torch.int64):
                x = torch.tensor(
                    [[9, 10, 17], [18, 100, 102]], dtype=dtype, device=device
                )
                for tokens in (x, x.t(), x[:, ::2]):
                    with self.subTest(
                        device=device, dtype=dtype, strides=tokens.stride()
                    ):
                        self.check_case(tokens, cfg)

    def test_current_stream(self):
        # The producer and consumer share a non-default stream. No host
        # readback or global synchronization is needed between the operations.
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            x = torch.arange(32, dtype=torch.int32, device="npu") + 10
            mapped, mask = embedding.get_masked_input_and_mask(x, 10, 42, 0, 100, 100)
            consumed = mapped + 1
        stream.synchronize()
        self.assertEqual(consumed.cpu().tolist(), list(range(1, 33)))
        self.assertFalse(mask.cpu().any())


if __name__ == "__main__":
    unittest.main()
