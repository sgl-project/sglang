#!/usr/bin/env python3

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GROUP_SIZE = 16

# Written out independently of the implementation: the E2M1 code points in
# magnitude order, so index == the 3-bit magnitude code.
_REFERENCE_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def reference_dequant(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Comparison oracle. Kept as a plain per-element loop on purpose: a
    vectorized rewrite would mirror the code under test."""
    rows, half = packed.shape
    hidden = half * 2
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            byte = int(packed[r, c // 2])
            code = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
            magnitude = _REFERENCE_E2M1[code & 0x7]
            value = -magnitude if code & 0x8 else magnitude
            scale = float(block_scale[r, c // GROUP_SIZE]) * global_scale
            out[r, c] = value * scale
    return out


def build_layer(method, vocab_size: int, hidden_size: int) -> torch.nn.Module:
    """Materialize through create_weights, then fill as a checkpoint would."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[vocab_size],
        input_size=hidden_size,
        output_size=vocab_size,
        params_dtype=torch.bfloat16,
    )

    generator = torch.Generator().manual_seed(0)
    layer.weight.data.copy_(
        torch.randint(
            0,
            256,
            (vocab_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
    )
    # Keep the block scales in a range e4m3 represents exactly.
    layer.weight_scale.data.copy_(
        torch.randint(
            1,
            8,
            (vocab_size, hidden_size // GROUP_SIZE),
            dtype=torch.int32,
            generator=generator,
        ).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.data.fill_(0.125)
    return layer


class TestNvFp4Embedding(CustomTestCase):
    def setUp(self):
        self.method = ModelOptNvFp4EmbeddingMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE
            )
        )

    def test_matches_reference_dequant(self):
        vocab_size, hidden_size = 24, 64
        layer = build_layer(self.method, vocab_size, hidden_size)
        self.assertEqual(tuple(layer.weight.shape), (vocab_size, hidden_size // 2))
        self.assertEqual(
            tuple(layer.weight_scale.shape), (vocab_size, hidden_size // GROUP_SIZE)
        )

        ids = torch.tensor([[0, 5, 5], [23, 11, 0]])
        got = self.method.embedding(layer, ids)
        expected = reference_dequant(
            layer.weight[ids.reshape(-1)],
            layer.weight_scale[ids.reshape(-1)].float(),
            float(layer.weight_scale_2),
        )

        self.assertEqual(tuple(got.shape), (2, 3, hidden_size))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(
            got.reshape(-1, hidden_size).float(),
            expected.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
        )

    def test_hidden_size_must_divide_group_size(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            self.method.create_weights(
                torch.nn.Module(),
                input_size_per_partition=40,
                output_partition_sizes=[8],
                input_size=40,
                output_size=8,
                params_dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
