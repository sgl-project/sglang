"""Engram projection topology guards and unquantized weight partitioning."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.engram import build_engram_projection
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEngramProjection(CustomTestCase):
    def build(self, **changes):
        options = dict(
            enabled=True,
            role="prefill",
            tp_rank=0,
            tp_size=8,
            quant_config=None,
            output_size=256,
        )
        options.update(changes)
        return build_engram_projection(
            64, prefix="model.layers.1.engram.wkv", **options
        )

    def test_sharded_weight_loader_reconstructs_original(self):
        full = torch.arange(256 * 64, dtype=torch.float32).reshape(256, 64)
        pieces = []
        for rank in range(8):
            layer = self.build(tp_rank=rank)
            layer.weight.weight_loader(layer.weight, full)
            self.assertEqual(tuple(layer.weight.shape), (32, 64))
            self.assertTrue(layer.gather_output)
            pieces.append(layer.weight.detach())
        torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=0)

    def test_disabled_keeps_full_weight(self):
        layer = self.build(enabled=False, role="decode")
        self.assertEqual(tuple(layer.weight.shape), (256, 64))

    def test_unsafe_topologies_are_rejected(self):
        for change in [
            dict(role="decode"),
            dict(role="null"),
            dict(dp_attention=True),
            dict(cp_size=2),
            dict(prefill_cp=True),
            dict(sequence_parallel=True),
            dict(tp_size=4),
        ]:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "requires Prefill, TP8"),
            ):
                self.build(**change)

    def test_unaligned_partitions_are_rejected(self):
        for change in [
            dict(output_size=257),
            dict(
                output_size=264,
                quant_config=SimpleNamespace(weight_block_size=[32, 32]),
            ),
        ]:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(
                    ValueError, "align with weight quantization blocks"
                ),
            ):
                self.build(**change)


if __name__ == "__main__":
    unittest.main()
