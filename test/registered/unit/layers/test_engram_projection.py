"""Engram projection topology guards and unquantized weight partitioning."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.engram import build_engram_projection
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEngramProjection(CustomTestCase):
    def build(self, **changes):
        options = dict(
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
        for role, tp_size in (
            ("prefill", 4),
            ("prefill", 8),
            ("decode", 4),
            ("decode", 8),
            ("null", 4),
            ("null", 8),
        ):
            with self.subTest(role=role, tp_size=tp_size):
                pieces = []
                for rank in range(tp_size):
                    layer = self.build(role=role, tp_rank=rank, tp_size=tp_size)
                    layer.weight.weight_loader(layer.weight, full)
                    self.assertEqual(tuple(layer.weight.shape), (256 // tp_size, 64))
                    self.assertIsInstance(layer, ColumnParallelLinear)
                    self.assertTrue(layer.gather_output)
                    pieces.append(layer.weight.detach())
                torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=0)

    def test_unsupported_topologies_keep_full_weight(self):
        for change in [
            dict(role=None),
            dict(role="invalid"),
            dict(dp_attention=True),
            dict(cp_size=2),
            dict(prefill_cp=True),
            dict(sequence_parallel=True),
            dict(tp_size=1),
            dict(tp_size=2),
            dict(tp_size=16),
        ]:
            with self.subTest(change=change):
                layer = self.build(**change)
                self.assertIsInstance(layer, ReplicatedLinear)
                self.assertEqual(tuple(layer.weight.shape), (256, 64))

    def test_unaligned_output_keeps_full_weight(self):
        layer = self.build(output_size=257)
        self.assertIsInstance(layer, ReplicatedLinear)
        self.assertEqual(tuple(layer.weight.shape), (257, 64))

    def test_quantization_block_alignment_selects_projection(self):
        # Check layer selection independently of a GPU quantization backend.
        quant_config = SimpleNamespace(weight_block_size=[32, 32])
        for tp_size in (4, 8):
            for output_size in (256, 264):
                with self.subTest(tp_size=tp_size, output_size=output_size):
                    with (
                        patch(
                            "sglang.srt.layers.engram.ReplicatedLinear"
                        ) as replicated,
                        patch(
                            "sglang.srt.layers.engram.ColumnParallelLinear"
                        ) as sharded,
                    ):
                        self.build(
                            tp_size=tp_size,
                            output_size=output_size,
                            quant_config=quant_config,
                        )
                        selected, unused = (
                            (sharded, replicated)
                            if output_size == 256
                            else (replicated, sharded)
                        )
                        selected.assert_called_once()
                        unused.assert_not_called()
                        self.assertIs(
                            selected.call_args.kwargs["quant_config"], quant_config
                        )


if __name__ == "__main__":
    unittest.main()
