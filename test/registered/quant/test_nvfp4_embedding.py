#!/usr/bin/env python3
"""Unit tests for NVFP4 token-embedding dequantization on gather."""

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
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: float,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Naive per-element NVFP4 dequant used as the comparison oracle.

    Deliberately a plain Python loop over the packed nibbles rather than a
    vectorized rewrite of the code under test.
    """
    rows, half = packed.shape
    hidden = half * 2
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            byte = int(packed[r, c // 2])
            code = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
            magnitude = _REFERENCE_E2M1[code & 0x7]
            value = -magnitude if code & 0x8 else magnitude
            scale = float(block_scale[r, c // group_size]) * global_scale
            out[r, c] = value * scale
    return out


def make_method(group_size: int = GROUP_SIZE) -> ModelOptNvFp4EmbeddingMethod:
    config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True, group_size=group_size
    )
    return ModelOptNvFp4EmbeddingMethod(config)


def build_layer(
    method: ModelOptNvFp4EmbeddingMethod,
    vocab_size: int,
    hidden_size: int,
    seed: int = 0,
) -> torch.nn.Module:
    """Materialize the layer through create_weights, then fill the packed
    tensors the way a checkpoint would."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[vocab_size],
        input_size=hidden_size,
        output_size=vocab_size,
        params_dtype=torch.bfloat16,
    )

    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0, 256, (vocab_size, hidden_size // 2), dtype=torch.uint8, generator=generator
    )
    # Keep the block scales in a range e4m3 represents exactly.
    scale = torch.randint(
        1,
        8,
        (vocab_size, hidden_size // GROUP_SIZE),
        dtype=torch.int32,
        generator=generator,
    ).to(torch.float8_e4m3fn)

    layer.weight.data.copy_(packed)
    layer.weight_scale.data.copy_(scale)
    layer.weight_scale_2.data.fill_(0.125)
    return layer


class TestNvFp4EmbeddingWeights(CustomTestCase):
    def test_create_weights_shapes_and_dtypes(self):
        method = make_method()
        layer = build_layer(method, vocab_size=32, hidden_size=64)

        self.assertEqual(tuple(layer.weight.shape), (32, 32))
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.weight_scale.shape), (32, 4))
        self.assertEqual(layer.weight_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight_scale_2.shape), (1,))
        self.assertEqual(layer.weight_scale_2.dtype, torch.float32)

    def test_hidden_size_must_divide_group_size(self):
        method = make_method()
        layer = torch.nn.Module()
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            method.create_weights(
                layer,
                input_size_per_partition=40,
                output_partition_sizes=[8],
                input_size=40,
                output_size=8,
                params_dtype=torch.bfloat16,
            )

    def test_apply_is_rejected(self):
        method = make_method()
        with self.assertRaisesRegex(NotImplementedError, "gather-only"):
            method.apply(torch.nn.Module(), torch.zeros(1))


class TestNvFp4EmbeddingGather(CustomTestCase):
    def test_matches_reference_dequant(self):
        method = make_method()
        vocab_size, hidden_size = 24, 64
        layer = build_layer(method, vocab_size, hidden_size)

        ids = torch.tensor([0, 5, 5, 23, 11])
        got = method.embedding(layer, ids)

        expected_rows = reference_dequant(
            layer.weight[ids],
            layer.weight_scale[ids].float(),
            float(layer.weight_scale_2),
        )
        self.assertEqual(tuple(got.shape), (ids.numel(), hidden_size))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(
            got.float(), expected_rows.to(torch.bfloat16).float(), rtol=0, atol=0
        )

    def test_preserves_index_shape(self):
        method = make_method()
        layer = build_layer(method, vocab_size=16, hidden_size=32)

        ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        got = method.embedding(layer, ids)
        self.assertEqual(tuple(got.shape), (2, 3, 32))

        flat = method.embedding(layer, ids.reshape(-1))
        torch.testing.assert_close(got.reshape(-1, 32).float(), flat.float())

    def test_repeated_ids_gather_identical_rows(self):
        method = make_method()
        layer = build_layer(method, vocab_size=16, hidden_size=32)

        got = method.embedding(layer, torch.tensor([7, 7]))
        torch.testing.assert_close(got[0].float(), got[1].float())


if __name__ == "__main__":
    unittest.main(verbosity=2)
