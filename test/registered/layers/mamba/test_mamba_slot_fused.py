"""Unit tests for the fused conv-slot clear/copy kernels
(srt/mem_cache/mamba_slot_fused.py), checked bit-exact against the per-tensor
reference loop that MambaPool.clear_slots / copy_from fall back to.

Covers heterogeneous conv shapes, single- and multi-layer pools, single /
partial / full index sets, int32 indices, and the strided per-slot-envelope
layout used by page-major / unified pools.
"""

import unittest

import torch

from sglang.srt.mem_cache.mamba_slot_fused import (
    build_conv_slot_descriptor,
    fused_clear_conv_slots,
    fused_copy_conv_slots,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = get_device()

CONV_LEN = 3
# Representative hybrid conv-state trailing dims (a couple of KV-projection
# streams + wider residual streams); only the trailing dim differs.
HETERO_DIMS = [128, 128, 256, 256, 6144, 6144]

# (dims, num_layers, pool_size)
CONFIGS = [
    (HETERO_DIMS, 1, 460),  # single-layer draft pool, realistic size
    (HETERO_DIMS, 3, 128),  # multi-layer
    ([128], 1, 64),  # single conv tensor
    ([256, 6144], 2, 32),  # mixed shapes, 2 layers
]


def _make_convs(dims, num_layers, pool, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return [
        torch.randn(
            num_layers,
            pool,
            CONV_LEN,
            d,
            dtype=torch.bfloat16,
            device=device,
            generator=g,
        )
        for d in dims
    ]


def _envelope_views(buf, dims, num_layers, pool):
    """Strided per-slot-envelope views (page-major / unified layout): each conv
    tensor is a slice inside a shared per-slot entry, so slot_stride is the whole
    envelope, not the feature length."""
    envelope = buf.shape[2]
    views, off = [], 0
    for d in dims:
        views.append(
            torch.as_strided(
                buf,
                size=(num_layers, pool, CONV_LEN, d),
                stride=(pool * envelope, envelope, d, 1),
                storage_offset=off,
            )
        )
        off += CONV_LEN * d
    return views


def _ref_clear(convs, idx):
    for t in convs:
        t[:, idx] = 0


def _ref_copy(convs, src, dst):
    for t in convs:
        t[:, dst] = t[:, src]


@unittest.skipUnless(
    DEVICE in ("cuda", "xpu"), "fused conv-slot Triton kernels need CUDA or XPU"
)
class TestMambaSlotFused(CustomTestCase):
    def test_clear_matches_reference(self):
        dev = DEVICE
        for dims, num_layers, pool in CONFIGS:
            for n in sorted({1, pool // 3, pool}):  # single / partial / all slots
                with self.subTest(dims=dims, num_layers=num_layers, pool=pool, n=n):
                    base = _make_convs(dims, num_layers, pool, dev, seed=0)
                    idx = torch.randperm(pool, device=dev)[:n].to(torch.int64)
                    ref = [t.clone() for t in base]
                    got = [t.clone() for t in base]
                    _ref_clear(ref, idx)
                    fused_clear_conv_slots(build_conv_slot_descriptor(got), idx)
                    torch.get_device_module(dev).synchronize()
                    for r, g in zip(ref, got):
                        self.assertTrue(torch.equal(r, g))
                    # Cleared slots are exactly zero; the rest is untouched.
                    keep = torch.ones(pool, dtype=torch.bool, device=dev)
                    keep[idx] = False
                    for g, b in zip(got, base):
                        self.assertTrue((g[:, idx] == 0).all().item())
                        self.assertTrue(torch.equal(g[:, keep], b[:, keep]))

    def test_copy_matches_reference(self):
        dev = DEVICE
        for dims, num_layers, pool in CONFIGS:
            with self.subTest(dims=dims, num_layers=num_layers, pool=pool):
                base = _make_convs(dims, num_layers, pool, dev, seed=1)
                perm = torch.randperm(pool, device=dev)
                n = max(1, pool // 4)
                src = perm[:n].to(torch.int64)  # disjoint from dst (COW invariant)
                dst = perm[n : 2 * n].to(torch.int64)
                ref = [t.clone() for t in base]
                got = [t.clone() for t in base]
                _ref_copy(ref, src, dst)
                fused_copy_conv_slots(build_conv_slot_descriptor(got), src, dst)
                torch.get_device_module(dev).synchronize()
                for r, g in zip(ref, got):
                    self.assertTrue(torch.equal(r, g))

    def test_strided_envelope_layout(self):
        # Page-major / unified pools store conv tensors as strided views into a
        # shared per-slot envelope (slot_stride = whole entry >> feat). The
        # kernel reads real strides, so it must handle this; the whole envelope
        # buffer (including the other streams' bytes in each slot) must be
        # bit-exact vs the reference, proving no cross-stream clobber.
        dev = DEVICE
        num_layers, pool = 2, 48
        dims = [128, 256, 6144]
        envelope = sum(CONV_LEN * d for d in dims)
        g = torch.Generator(device=dev).manual_seed(4)
        buf = torch.randn(
            num_layers, pool, envelope, dtype=torch.bfloat16, device=dev, generator=g
        )
        v0 = _envelope_views(buf, dims, num_layers, pool)[0]
        self.assertFalse(v0.is_contiguous())  # strided view...
        self.assertTrue(v0[0, 0].is_contiguous())  # ...but per-slot block is not

        idx = torch.tensor([2, 7, 40], dtype=torch.int64, device=dev)
        ref_buf = buf.clone()
        got_buf = buf.clone()
        _ref_clear(_envelope_views(ref_buf, dims, num_layers, pool), idx)
        fused_clear_conv_slots(
            build_conv_slot_descriptor(
                _envelope_views(got_buf, dims, num_layers, pool)
            ),
            idx,
        )
        torch.get_device_module(dev).synchronize()
        self.assertTrue(torch.equal(ref_buf, got_buf))

        # copy on the same strided layout
        src = torch.tensor([1, 20], dtype=torch.int64, device=dev)
        dst = torch.tensor([30, 45], dtype=torch.int64, device=dev)
        ref_buf = buf.clone()
        got_buf = buf.clone()
        _ref_copy(_envelope_views(ref_buf, dims, num_layers, pool), src, dst)
        fused_copy_conv_slots(
            build_conv_slot_descriptor(
                _envelope_views(got_buf, dims, num_layers, pool)
            ),
            src,
            dst,
        )
        torch.get_device_module(dev).synchronize()
        self.assertTrue(torch.equal(ref_buf, got_buf))

    def test_empty_indices_is_noop(self):
        dev = DEVICE
        base = _make_convs(HETERO_DIMS, 1, 16, dev, seed=2)
        got = [t.clone() for t in base]
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        desc = build_conv_slot_descriptor(got)
        fused_clear_conv_slots(desc, empty)
        fused_copy_conv_slots(desc, empty, empty)
        torch.get_device_module(dev).synchronize()
        for b, g in zip(base, got):
            self.assertTrue(torch.equal(b, g))

    def test_int32_indices_accepted(self):
        # deferred-clear/COW indices are staged as int32; the wrappers must upcast.
        dev = DEVICE
        base = _make_convs(HETERO_DIMS, 1, 32, dev, seed=3)
        idx = torch.tensor([1, 5, 9], dtype=torch.int32, device=dev)
        ref = [t.clone() for t in base]
        got = [t.clone() for t in base]
        _ref_clear(ref, idx.long())
        fused_clear_conv_slots(build_conv_slot_descriptor(got), idx)
        torch.get_device_module(dev).synchronize()
        for r, g in zip(ref, got):
            self.assertTrue(torch.equal(r, g))


if __name__ == "__main__":
    unittest.main()
