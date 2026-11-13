"""Regression tests for the int32 page-offset overflow in ``_page_split_kernel``.

``_page_split_kernel`` derives ``page_idx``/``sub`` from ``tl.program_id(0)``,
which is int32, and then forms two independent byte offsets from it::

    src_base = src_ptr + page_idx * src_stride0
    dst_base = dst_ptr + (page_idx * RATIO + sub) * dst_stride0

Each product wraps once it crosses 2**31, and Triton sign-extends the wrapped
value onto the 64-bit pointer, putting the access ~2 GiB outside the intended
buffer -- a stray write into a neighbouring allocation with no error signal, or
an illegal access if that page is unmapped.

The two offsets cross the boundary at different pool sizes, so both are pinned
here rather than relying on one to imply the other:

* the destination side wraps first, at dst page index 57,359, i.e. once the pool
  holds 14,340 source pages (3,671,040 tokens/rank);
* the source side wraps one page later, at ``page_idx`` 14,340 with the shipped
  149,760 B page stride.

Ordinary (non-wrapping) split behaviour -- masked page skipping, the alignment
tail, the persistent-buffer contract -- is covered by ``TestTouchedPageSplit``
in ``test_flash_mla_backends.py``, which drives the production entry point.
These tests only add what that one cannot reach: the arithmetic past 2**31.

Keeping this affordable: ``src_stride0`` and ``dst_stride0`` are both kernel
parameters, so each test packs the axis it is *not* exercising and only pays for
the one it is. A buffer that genuinely spans past 2**31 bytes is irreducible --
that is the whole point -- but only one of them has to at a time, so peak
footprint stays near 4 GiB rather than the ~8 GiB a combined run would need.

Each oversized buffer is carved out of a larger allocation whose leading region
is a canary. A regression then lands inside memory this test owns, and surfaces
as an assertion rather than an illegal access that would poison the CUDA context
and cascade into unrelated tests sharing the process.
"""

import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _BYTES_PER_DST_PAGE_PADDED,
    _NOPE_ROPE_STRIDE,
    _PBS_DST,
    _PBS_SRC,
    _SCALE_STRIDE,
    _page_split_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

RATIO = _PBS_SRC // _PBS_DST
DATA_PER_SUB = _PBS_DST * _NOPE_ROPE_STRIDE
SCALE_PER_SUB = _PBS_DST * _SCALE_STRIDE
SRC_SCALE_OFF = _PBS_SRC * _NOPE_ROPE_STRIDE
DST_SCALE_OFF = _PBS_DST * _NOPE_ROPE_STRIDE
SRC_PAGE_FOOTPRINT = SRC_SCALE_OFF + RATIO * SCALE_PER_SUB  # bytes actually read
BLOCK_SIZE = 1024

PACKED_SRC_STRIDE = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 584; pages may overlap

INT32_MAX = 2**31 - 1
CANARY_BYTES = 2 * 1024**3
CANARY_FILL = 0xAB

# Signatures are per (page, sub) so a dropped sub-page offset is visible, and
# are confined to ranges that exclude both 0 (the destination's initial fill,
# which would make "kernel wrote nothing" indistinguishable from success) and
# CANARY_FILL (which a wrapped *read* would pull in).
_DATA_SIG_LO, _DATA_SIG_SPAN = 1, 100  # 1..100
_SCALE_SIG_LO, _SCALE_SIG_SPAN = 120, 40  # 120..159

FIRST_WRAPPING_DST_PID = INT32_MAX // _BYTES_PER_DST_PAGE_PADDED + 1  # 57,359
PEAK_BYTES = CANARY_BYTES + 2 * 1024**3 + (1 << 28)  # worst case across tests


def _launch(src, dst, n_pages, src_stride, mask, has_mask):
    _page_split_kernel[(n_pages * RATIO,)](
        src,
        dst,
        n_pages,
        src_stride,
        _BYTES_PER_DST_PAGE_PADDED,
        DATA_PER_SUB,
        SCALE_PER_SUB,
        SRC_SCALE_OFF,
        DST_SCALE_OFF,
        RATIO,
        BLOCK_SIZE,
        mask,
        has_mask,
    )
    torch.cuda.synchronize()


def _fill_signed_src(src, n_pages):
    """Give every (page, sub-page) its own byte signature.

    Returns the per-destination-page expected data and scale bytes, indexed by
    ``pid`` (= ``page_idx * RATIO + sub``), which is exactly what the kernel
    should deposit in destination page ``pid``.
    """
    device = src.device
    pid = torch.arange(n_pages * RATIO, device=device, dtype=torch.int32)
    data_sig = (_DATA_SIG_LO + (pid * 7) % _DATA_SIG_SPAN).to(torch.uint8)
    scale_sig = (_SCALE_SIG_LO + (pid * 11) % _SCALE_SIG_SPAN).to(torch.uint8)
    for sub in range(RATIO):
        lo = sub * DATA_PER_SUB
        src[:, lo : lo + DATA_PER_SUB] = data_sig[sub::RATIO][:, None]
        lo = SRC_SCALE_OFF + sub * SCALE_PER_SUB
        src[:, lo : lo + SCALE_PER_SUB] = scale_sig[sub::RATIO][:, None]
    return data_sig, scale_sig


class TestPageSplitInt32Offset(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        free, _ = torch.cuda.mem_get_info()
        if free < PEAK_BYTES + (1 << 30):
            raise unittest.SkipTest(
                f"needs ~{(PEAK_BYTES + (1 << 30)) / 1024**3:.1f} GiB free, "
                f"have {free / 1024**3:.1f} GiB"
            )

    def tearDown(self):
        torch.cuda.empty_cache()

    def _assert_canary_intact(self, canary, what):
        # min/max reductions avoid materializing a 2 GiB comparison mask.
        chunk = 1 << 26
        for start in range(0, canary.numel(), chunk):
            seg = canary[start : start + chunk]
            self.assertEqual(
                (int(seg.min()), int(seg.max())),
                (CANARY_FILL, CANARY_FILL),
                f"stray write into memory below the {what} buffer "
                f"(canary chunk at +{start}): the page-split offset wrapped",
            )

    def test_destination_offset_does_not_wrap_past_int32(self):
        """dst_base = (page_idx * RATIO + sub) * dst_stride0 beyond 2**31."""
        device = torch.device("cuda")
        n_pages = FIRST_WRAPPING_DST_PID // RATIO + 1  # 14,340
        grid = n_pages * RATIO
        boundary = n_pages - 1
        self.assertLess(
            FIRST_WRAPPING_DST_PID,
            grid,
            "grid must reach past the int32 boundary for this test to mean anything",
        )

        # Source packed tight: this test exercises the destination axis only.
        # Random bytes rather than page signatures, because packed pages overlap
        # and a per-page fill would alias; randomness also varies within each
        # sub-page, so a dropped data_src_off/scale_src_off shows up here.
        src_bytes = boundary * PACKED_SRC_STRIDE + SRC_PAGE_FOOTPRINT
        src = torch.randint(0, 256, (src_bytes,), dtype=torch.uint8, device=device)
        big = torch.empty(
            CANARY_BYTES + grid * _BYTES_PER_DST_PAGE_PADDED,
            dtype=torch.uint8,
            device=device,
        )
        dst = big[CANARY_BYTES:].view(grid, _BYTES_PER_DST_PAGE_PADDED)
        mask = torch.zeros(n_pages, dtype=torch.int8, device=device)
        mask[boundary] = 1

        for has_mask in (True, False):
            with self.subTest(has_mask=has_mask):
                dst.zero_()
                big[:CANARY_BYTES].fill_(CANARY_FILL)
                _launch(src, dst, n_pages, PACKED_SRC_STRIDE, mask, has_mask)

                base = boundary * PACKED_SRC_STRIDE
                for sub in range(RATIO):
                    pid = boundary * RATIO + sub
                    lo = base + sub * DATA_PER_SUB
                    self.assertTrue(
                        torch.equal(
                            dst[pid, :DATA_PER_SUB], src[lo : lo + DATA_PER_SUB]
                        ),
                        f"data region of boundary dst page {pid} is wrong",
                    )
                    lo = base + SRC_SCALE_OFF + sub * SCALE_PER_SUB
                    self.assertTrue(
                        torch.equal(
                            dst[pid, DST_SCALE_OFF : DST_SCALE_OFF + SCALE_PER_SUB],
                            src[lo : lo + SCALE_PER_SUB],
                        ),
                        f"scale region of boundary dst page {pid} is wrong",
                    )
                self._assert_canary_intact(big[:CANARY_BYTES], "destination")

    def test_source_offset_does_not_wrap_past_int32(self):
        """src_base = page_idx * src_stride0 beyond 2**31.

        Isolated from the destination axis by inflating ``src_stride0`` so the
        source crosses the boundary at a tiny page count, which keeps the
        destination allocation at a few MB. The canary sits below the source, so
        a wrapped *read* pulls canary bytes into the destination instead of
        faulting -- caught by the signature comparison.
        """
        device = torch.device("cuda")
        src_stride = 1 << 27  # 134,217,728: page 16 lands exactly on 2**31
        first_wrapping_page = INT32_MAX // src_stride + 1
        n_pages = first_wrapping_page + 1
        grid = n_pages * RATIO
        self.assertGreaterEqual(src_stride, SRC_PAGE_FOOTPRINT)
        self.assertGreater(
            first_wrapping_page * src_stride,
            INT32_MAX,
            "chosen stride must actually push the source offset past int32",
        )

        big = torch.empty(
            CANARY_BYTES + n_pages * src_stride, dtype=torch.uint8, device=device
        )
        big[:CANARY_BYTES].fill_(CANARY_FILL)
        src = big[CANARY_BYTES:].view(n_pages, src_stride)
        data_sig, scale_sig = _fill_signed_src(src, n_pages)
        dst = torch.zeros(
            (grid, _BYTES_PER_DST_PAGE_PADDED), dtype=torch.uint8, device=device
        )
        mask = torch.ones(n_pages, dtype=torch.int8, device=device)

        _launch(src, dst, n_pages, src_stride, mask, False)

        for pid in range(grid):
            with self.subTest(pid=pid, page_idx=pid // RATIO):
                self.assertTrue(
                    bool((dst[pid, :DATA_PER_SUB] == data_sig[pid]).all()),
                    f"data region of dst page {pid} is wrong",
                )
                self.assertTrue(
                    bool(
                        (
                            dst[pid, DST_SCALE_OFF : DST_SCALE_OFF + SCALE_PER_SUB]
                            == scale_sig[pid]
                        ).all()
                    ),
                    f"scale region of dst page {pid} is wrong",
                )


if __name__ == "__main__":
    unittest.main()
