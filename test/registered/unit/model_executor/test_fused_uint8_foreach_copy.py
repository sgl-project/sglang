"""Unit tests for ``_fused_uint8_foreach_copy_``.

The helper reinterprets each foldable (dst, src) pair as ``uint8`` so a single
``torch._foreach_copy_`` fuses all dtypes into one batched memcpy. A pair is
foldable only when it shares dtype + shape + device and both sides are
contiguous; anything else defers to ``copy_()`` (which casts / walks strides).

These tests verify:
  * byte-equivalence with per-tensor ``copy_()`` across mixed dtypes/ranks,
  * raw-byte fidelity for bf16 (incl. NaN / -0.0 bit patterns),
  * the foldability guard routes dtype-mismatch / non-contiguous /
    shape-mismatch / cross-device pairs to ``copy_()``,
  * the dispatch actually fuses (one ``_foreach_copy_`` over only the
    foldable pairs),
  * the no-``_foreach_copy_`` fallback path.

Registered on both CPU and CUDA: the dtype/shape/contiguity logic is
device-independent and runs on the CPU runner, while the cross-device and
device-to-device fold cases only execute on the GPU runner (they are
``skipUnless(torch.cuda.is_available())`` and would silently never run if this
file were CPU-only).
"""

import unittest
import unittest.mock as mock

import torch

from sglang.srt.model_executor.runner_utils import buffers
from sglang.srt.model_executor.runner_utils.buffers import _fused_uint8_foreach_copy_
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two same-dtype tensors bit-for-bit (NaN / signed-zero safe)."""
    assert a.dtype == b.dtype and a.shape == b.shape
    n = a.element_size()
    int_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[n]
    return torch.equal(
        a.contiguous().view(-1).view(int_dtype),
        b.contiguous().view(-1).view(int_dtype),
    )


class TestFusedUint8ForeachCopy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # -- correctness ---------------------------------------------------------

    def test_fold_mixed_dtypes_and_ranks_matches_copy(self):
        srcs = [
            torch.randint(0, 1000, (5,), dtype=torch.int64),
            torch.randn(5, 7, dtype=torch.float32),
            torch.randn(5, 4).to(torch.bfloat16),
            torch.randint(0, 8, (5,), dtype=torch.int32),
            torch.randint(0, 50, (3, 2, 4), dtype=torch.int64),  # 3-D
        ]
        dsts = [torch.zeros_like(s) for s in srcs]
        ref = [torch.zeros_like(s) for s in srcs]
        for d, s in zip(ref, srcs):
            d.copy_(s)

        _fused_uint8_foreach_copy_(dsts, srcs)

        for d, r in zip(dsts, ref):
            self.assertEqual(d.dtype, r.dtype)
            self.assertEqual(d.shape, r.shape)
            self.assertTrue(torch.equal(d, r))

    def test_bf16_raw_byte_fidelity(self):
        # Exact bit patterns (NaN / inf / -0.0) must survive the uint8 byte copy
        # — a value-compare can't even express NaN equality, so compare bits.
        special = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0],
            dtype=torch.bfloat16,
        )
        src = torch.cat([special, torch.randn(11).to(torch.bfloat16)])
        dst = torch.zeros_like(src)

        _fused_uint8_foreach_copy_([dst], [src])

        self.assertTrue(_bits_equal(dst, src))

    def test_scalar_tensors_fold(self):
        dst = torch.zeros((), dtype=torch.int64)
        src = torch.tensor(42, dtype=torch.int64)
        _fused_uint8_foreach_copy_([dst], [src])
        self.assertEqual(int(dst.item()), 42)

    def test_source_not_mutated(self):
        srcs = [torch.randn(4, 3), torch.randint(0, 9, (4,), dtype=torch.int64)]
        before = [s.clone() for s in srcs]
        dsts = [torch.zeros_like(s) for s in srcs]
        _fused_uint8_foreach_copy_(dsts, srcs)
        for s, b in zip(srcs, before):
            self.assertTrue(torch.equal(s, b))

    def test_empty_lists_noop(self):
        # Must not raise.
        _fused_uint8_foreach_copy_([], [])

    # -- foldability guard / fallbacks --------------------------------------

    def test_dtype_mismatch_falls_back_to_cast(self):
        # int32 -> int64 must cast (a raw byte view would corrupt).
        src = torch.randint(0, 1000, (6,), dtype=torch.int32)
        dst = torch.zeros(6, dtype=torch.int64)
        _fused_uint8_foreach_copy_([dst], [src])
        self.assertTrue(torch.equal(dst, src.to(torch.int64)))

    def test_noncontiguous_src_falls_back(self):
        base = torch.randn(5, 16)
        src = base[:, ::2]  # strided, non-contiguous view, shape [5, 8]
        self.assertFalse(src.is_contiguous())
        dst = torch.zeros(5, 8)
        _fused_uint8_foreach_copy_([dst], [src])
        self.assertTrue(torch.equal(dst, src))

    def test_shape_mismatch_same_numel_raises_like_copy(self):
        # Same numel, non-broadcastable shapes: copy_ raises -> the helper must
        # raise too (NOT silently byte-copy transposed data).
        dst = torch.zeros(2, 3)
        src = torch.arange(6.0).reshape(3, 2)

        copy_raised = False
        try:
            torch.zeros(2, 3).copy_(src)
        except RuntimeError:
            copy_raised = True
        self.assertTrue(copy_raised, "precondition: copy_ should reject this")

        with self.assertRaises(RuntimeError):
            _fused_uint8_foreach_copy_([dst], [src])

    def test_mixed_foldable_and_fallback(self):
        # foldable (f32) + dtype-mismatch (i32->i64) + non-contiguous, one call.
        s_fold = torch.randn(4, 3)
        d_fold = torch.zeros_like(s_fold)
        s_cast = torch.randint(0, 9, (4,), dtype=torch.int32)
        d_cast = torch.zeros(4, dtype=torch.int64)
        s_nc = torch.randn(4, 8)[:, ::2]
        d_nc = torch.zeros(4, 4)

        _fused_uint8_foreach_copy_([d_fold, d_cast, d_nc], [s_fold, s_cast, s_nc])

        self.assertTrue(torch.equal(d_fold, s_fold))
        self.assertTrue(torch.equal(d_cast, s_cast.to(torch.int64)))
        self.assertTrue(torch.equal(d_nc, s_nc))

    # -- dispatch (does it actually fuse?) ----------------------------------

    def test_dispatch_fuses_all_foldable_into_one_foreach(self):
        srcs = [
            torch.randint(0, 9, (5,), dtype=torch.int64),
            torch.randn(5, 3),
            torch.randint(0, 9, (5,), dtype=torch.int32),
        ]
        dsts = [torch.zeros_like(s) for s in srcs]

        real = torch._foreach_copy_
        calls = []

        def spy(d, s, *a, **k):
            calls.append((list(d), list(s)))
            return real(d, s, *a, **k)

        with mock.patch.object(torch, "_foreach_copy_", side_effect=spy):
            _fused_uint8_foreach_copy_(dsts, srcs)

        # One fused call covering all 3 pairs, every tensor reinterpreted uint8.
        self.assertEqual(len(calls), 1)
        folded_dsts, folded_srcs = calls[0]
        self.assertEqual(len(folded_dsts), 3)
        self.assertTrue(all(t.dtype == torch.uint8 for t in folded_dsts))
        self.assertTrue(all(t.dtype == torch.uint8 for t in folded_srcs))
        # ...and the result is still correct.
        for d, s in zip(dsts, srcs):
            self.assertTrue(torch.equal(d, s))

    def test_dispatch_excludes_nonfoldable_pairs(self):
        s0 = torch.randn(4, 2)
        d0 = torch.zeros_like(s0)
        s1 = torch.randint(0, 9, (4,), dtype=torch.int32)  # cast -> fallback
        d1 = torch.zeros(4, dtype=torch.int64)
        s2 = torch.randint(0, 9, (4,), dtype=torch.int64)
        d2 = torch.zeros_like(s2)

        real = torch._foreach_copy_
        calls = []

        def spy(d, s, *a, **k):
            calls.append(list(d))
            return real(d, s, *a, **k)

        with mock.patch.object(torch, "_foreach_copy_", side_effect=spy):
            _fused_uint8_foreach_copy_([d0, d1, d2], [s0, s1, s2])

        # Only the 2 foldable pairs go through foreach; the cast pair uses copy_.
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertTrue(torch.equal(d0, s0))
        self.assertTrue(torch.equal(d1, s1.to(torch.int64)))
        self.assertTrue(torch.equal(d2, s2))

    def test_fallback_when_foreach_unavailable(self):
        srcs = [
            torch.randn(4, 3),
            torch.randint(0, 9, (4,), dtype=torch.int64),
            torch.randint(0, 9, (4,), dtype=torch.int32),
        ]
        dsts = [torch.zeros_like(s) for s in srcs]
        with mock.patch.object(buffers, "_has_foreach_copy", False):
            with mock.patch.object(
                torch, "_foreach_copy_", side_effect=AssertionError("must not call")
            ):
                _fused_uint8_foreach_copy_(dsts, srcs)
        for d, s in zip(dsts, srcs):
            self.assertTrue(torch.equal(d, s))

    # -- cross-device (CUDA only) -------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cross_device_pair_falls_back_to_copy(self):
        # Same dtype/shape but different device: must NOT enter the per-device
        # foreach kernel; copy_ handles the H2D transfer.
        src = torch.randint(0, 1000, (8,), dtype=torch.int64)  # CPU
        dst = torch.zeros(8, dtype=torch.int64, device="cuda")

        real = torch._foreach_copy_
        calls = []

        def spy(d, s, *a, **k):
            calls.append(list(d))
            return real(d, s, *a, **k)

        with mock.patch.object(torch, "_foreach_copy_", side_effect=spy):
            _fused_uint8_foreach_copy_([dst], [src])

        self.assertEqual(calls, [])  # cross-device pair was not fused
        self.assertTrue(torch.equal(dst.cpu(), src))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_d2d_fold_on_cuda(self):
        srcs = [
            torch.randint(0, 9, (5,), dtype=torch.int64, device="cuda"),
            torch.randn(5, 4, dtype=torch.float32, device="cuda"),
            torch.randn(5, 4, device="cuda").to(torch.bfloat16),
        ]
        dsts = [torch.zeros_like(s) for s in srcs]
        _fused_uint8_foreach_copy_(dsts, srcs)
        torch.cuda.synchronize()
        for d, s in zip(dsts, srcs):
            self.assertTrue(_bits_equal(d, s))


if __name__ == "__main__":
    unittest.main()
