# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Unit tests for CompressedTensorsEmbeddingMethod.

The dequant-on-gather logic for pack-quantized token embeddings is validated
against a deliberately-simple per-element oracle (no vectorized reuse of the
implementation under test).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsEmbeddingMethod,
)
from sglang.test.test_utils import CustomTestCase


def reference_dequant(
    packed: torch.Tensor, scale: torch.Tensor, num_bits: int, group_size: int
) -> torch.Tensor:
    """Plain per-element oracle. Kept as a loop on purpose."""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    half = 1 << (num_bits - 1)
    rows, packed_cols = packed.shape
    hidden = packed_cols * pack_factor
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            packed_idx = c // pack_factor
            shift = (c % pack_factor) * num_bits
            v = (int(packed[r, packed_idx]) >> shift) & mask
            q = v - half
            grp = c // group_size if group_size else 0
            out[r, c] = q * float(scale[r, grp])
    return out


def pack_values(vals: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack int values (rows, hidden) into int32 words, little-endian bit order."""
    pack_factor = 32 // num_bits
    rows, hidden = vals.shape
    mask = (1 << num_bits) - 1
    packed = torch.zeros(rows, hidden // pack_factor, dtype=torch.int32)
    for i in range(pack_factor):
        chunk = (vals[:, i::pack_factor] & mask).to(torch.int32)
        packed |= chunk << (i * num_bits)
    return packed


class TestCompressedTensorsEmbeddingMethod(CustomTestCase):
    def _make_method(self, num_bits: int, group_size: int):
        weight_quant = QuantizationArgs(
            num_bits=num_bits,
            group_size=group_size,
            strategy="group",
        )
        return CompressedTensorsEmbeddingMethod(weight_quant)

    def _make_layer(self, rows: int, hidden: int, num_bits: int, group_size: int):
        torch.manual_seed(0)
        qvals = torch.randint(0, 1 << num_bits, (rows, hidden))
        scale = torch.rand(rows, hidden // group_size) + 0.5
        packed = pack_values(qvals, num_bits)
        layer = type(
            "E",
            (),
            {
                "weight_packed": packed,
                "weight_scale": scale,
                "hidden_size": hidden,
            },
        )
        return layer

    def test_int8_group128(self):
        rows, hidden, num_bits, group_size = 8, 512, 8, 128
        layer = self._make_layer(rows, hidden, num_bits, group_size)
        method = self._make_method(num_bits, group_size)
        out = method.embedding(layer, torch.arange(rows))
        ref = reference_dequant(
            layer.weight_packed, layer.weight_scale, num_bits, group_size
        )
        torch.testing.assert_close(out, ref)

    def test_int4_group128(self):
        rows, hidden, num_bits, group_size = 8, 512, 4, 128
        layer = self._make_layer(rows, hidden, num_bits, group_size)
        method = self._make_method(num_bits, group_size)
        out = method.embedding(layer, torch.arange(rows))
        ref = reference_dequant(
            layer.weight_packed, layer.weight_scale, num_bits, group_size
        )
        torch.testing.assert_close(out, ref)

    def test_partial_gather(self):
        rows, hidden, num_bits, group_size = 8, 512, 8, 128
        layer = self._make_layer(rows, hidden, num_bits, group_size)
        method = self._make_method(num_bits, group_size)
        ids = torch.tensor([3, 1, 5])
        out = method.embedding(layer, ids)
        ref = reference_dequant(
            layer.weight_packed, layer.weight_scale, num_bits, group_size
        )[ids]
        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
