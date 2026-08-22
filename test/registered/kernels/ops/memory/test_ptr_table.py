"""Device-pointer tables must survive addresses with the top bit set.

Bug regression (issue #35047): these tables were built as ``int64``, so a base
address ``>= 2**63`` raised ``ValueError: Overflow when unpacking long long`` on
the host, before any launch. Backends whose addresses stay ``< 2**47`` cannot
catch this, so the cases below spoof a high address onto CPU tensors and run
anywhere.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.kernels.ops.memory.ptr_table import make_ptr_table
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        _conv_multi_build_meta,
    )
    from sglang.srt.mem_cache.mamba_slot_fused import build_conv_slot_descriptor

    _IMPORT_ERROR = None
except Exception as e:  # triton is not installed on every CPU runner
    _conv_multi_build_meta = None
    build_conv_slot_descriptor = None
    _IMPORT_ERROR = e

# A base address from the issue report; the top bit is set.
_HIGH_BASE = 0xFFFF85ABD4E00000


class _SpoofedPtrTensor(torch.Tensor):
    """A real tensor reporting a top-bit-set ``data_ptr()``; everything else
    (shape, strides, dtype, device) stays real."""

    _spoofed_ptr = None

    def data_ptr(self) -> int:
        if self._spoofed_ptr is None:
            return super().data_ptr()
        return self._spoofed_ptr


def _spoof(t: torch.Tensor, offset: int) -> torch.Tensor:
    spoofed = t.as_subclass(_SpoofedPtrTensor)
    spoofed._spoofed_ptr = _HIGH_BASE + offset
    return spoofed


def _conv_pairs(elems):
    """(dst, src) conv-window pairs as the multi-type scatter takes them."""
    layers, slots, batch, steps, dim = 2, 8, 4, 3, 1
    return [
        (
            torch.zeros((layers, slots, dim, e), dtype=torch.bfloat16),
            torch.zeros((layers, batch, steps, dim, e), dtype=torch.bfloat16),
        )
        for e in elems
    ]


def _conv_tensors(feats):
    """Conv-state pool tensors as ``build_conv_slot_descriptor`` takes them."""
    return [torch.zeros((2, 16, 4, f), dtype=torch.bfloat16) for f in feats]


class TestMakePtrTable(CustomTestCase):
    def test_full_range_addresses_round_trip(self):
        # Every 64-bit address survives the unsigned build and reads back
        # unchanged; 2**63 is the exact value an int64 build rejects.
        addrs = [0, 2**47, 2**63 - 1, 2**63, 2**64 - 1]
        table = make_ptr_table(addrs, device="cpu")
        self.assertEqual(table.dtype, torch.int64)
        self.assertEqual(table.view(torch.uint64).tolist(), addrs)


@unittest.skipUnless(_IMPORT_ERROR is None, f"import failed: {_IMPORT_ERROR}")
class TestPtrTableCallSites(CustomTestCase):
    """Each call site builds its table twice — from tensors reporting a
    top-bit-set address, and from the real low CPU addresses — asserting the
    address columns match what was reported and companion columns (strides,
    element counts, block offsets) are unchanged by the unsigned build."""

    def test_conv_multi_meta_table(self):
        elems = (128, 64)
        real = _conv_pairs(elems)
        spoofed = [
            (_spoof(dst, 2 * i * 4096), _spoof(src, (2 * i + 1) * 4096))
            for i, (dst, src) in enumerate(real)
        ]

        real_meta, real_blocks = _conv_multi_build_meta(real, block_size=64)
        meta, blocks = _conv_multi_build_meta(spoofed, block_size=64)

        self.assertEqual(meta.dtype, torch.int64)
        self.assertEqual(blocks, real_blocks)
        # Columns 0/1 are the src/dst base addresses.
        addrs = meta.view(torch.uint64)[:, :2].tolist()
        self.assertEqual(
            addrs, [[src.data_ptr(), dst.data_ptr()] for dst, src in spoofed]
        )
        self.assertTrue(torch.equal(meta[:, 2:], real_meta[:, 2:]))

    def test_conv_slot_descriptor(self):
        feats = (128, 6144)
        real = _conv_tensors(feats)
        spoofed = [_spoof(t, i * 4096) for i, t in enumerate(real)]

        real_desc = build_conv_slot_descriptor(real)
        desc = build_conv_slot_descriptor(spoofed)

        self.assertEqual(desc.ptr.dtype, torch.int64)
        self.assertEqual(
            desc.ptr.view(torch.uint64).tolist(), [t.data_ptr() for t in spoofed]
        )
        self.assertEqual(desc.num_layers, real_desc.num_layers)
        self.assertEqual(desc.max_feat_blocks, real_desc.max_feat_blocks)
        for field in ("feat", "layer_stride", "slot_stride"):
            self.assertTrue(
                torch.equal(getattr(desc, field), getattr(real_desc, field)), field
            )


if __name__ == "__main__":
    unittest.main()
