from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

import itertools
import unittest

import torch

from sglang.kernels.ops.speculative.gather_spec_extras import (
    gather_spec_extras,
    scatter_spec_extras,
)
from sglang.test.test_utils import CustomTestCase

_OUTPUT_NAMES = ("topk_p", "topk_index", "bonus_tokens", "hidden_states")


def _ref_gather(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    """Reference oracle: the exact torch.compile'd advanced-index gather that the
    fused Triton kernel replaced (see overlap_utils._gather_spec_extras pre-fusion).
    A gather is a pure copy, so the kernel must match this bit-for-bit."""
    topk_p = topk_p_buf[indices]
    topk_index = topk_index_buf[indices]
    bonus_tokens = output_tokens_buf[indices]
    hidden_states = (
        hidden_states_buf[indices] if hidden_states_buf is not None else None
    )
    return topk_p, topk_index, bonus_tokens, hidden_states


def _make_buffers(
    pool_size,
    topk,
    hidden_dim,
    *,
    with_hidden,
    hidden_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    """Build FutureMap-shaped relay buffers.

    Mirrors overlap_utils.FutureMap: topk_p / topk_index / hidden_states are
    2-D (pool_size, width) while output_tokens is 1-D (pool_size,). The width
    mix (incl. the 1-D buffer -> row width 1) exercises the kernel's per-buffer
    masking.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    topk_p_buf = torch.rand(
        (pool_size, topk), dtype=torch.float32, device=device, generator=g
    )
    topk_index_buf = torch.randint(
        0, 32000, (pool_size, topk), dtype=torch.int64, device=device, generator=g
    )
    output_tokens_buf = torch.randint(
        0, 32000, (pool_size,), dtype=torch.int64, device=device, generator=g
    )
    hidden_states_buf = (
        torch.randn(
            (pool_size, hidden_dim), dtype=hidden_dtype, device=device, generator=g
        )
        if with_hidden
        else None
    )
    return topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestGatherSpecExtras(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _assert_matches_reference(self, indices, bufs):
        """Run fused kernel + reference on the same inputs and assert every
        output is identical (dtype, shape, exact values) and that the source
        buffers are never mutated."""
        src_snapshots = [None if b is None else b.clone() for b in bufs]

        ref = _ref_gather(indices, *bufs)
        got = gather_spec_extras(indices, *bufs)

        self.assertEqual(len(got), len(ref))
        for name, r, o in zip(_OUTPUT_NAMES, ref, got):
            if r is None:
                self.assertIsNone(o, f"{name} should be None when buffer is None")
                continue
            self.assertIsNotNone(o, f"{name} unexpectedly None")
            self.assertEqual(o.dtype, r.dtype, f"{name} dtype mismatch")
            self.assertEqual(tuple(o.shape), tuple(r.shape), f"{name} shape mismatch")
            self.assertEqual(o.device.type, r.device.type, f"{name} device mismatch")
            # Pure gather == bit-exact copy, so require zero tolerance.
            torch.testing.assert_close(
                o, r, rtol=0, atol=0, msg=f"{name} value mismatch"
            )

        # The kernel only reads sources; it must not scribble into them.
        for name, before, buf in zip(_OUTPUT_NAMES, src_snapshots, bufs):
            if before is None:
                continue
            torch.testing.assert_close(
                buf, before, rtol=0, atol=0, msg=f"source buffer {name} was mutated"
            )

    def test_matches_reference_across_shapes(self):
        # (pool_size, m, topk, hidden_dim). Covers: m<pool, m==pool, topk==1,
        # 1-column blocks, exact-1024-width boundary, wide multi-block widths,
        # and wide non-power-of-2 widths (partial trailing column block).
        configs = [
            (16, 8, 1, 7),
            (64, 33, 4, 128),
            (128, 128, 8, 1024),
            (100, 50, 2, 4096),
            (257, 200, 16, 4097),
            (2048, 777, 8, 5120),
        ]
        for pool_size, m, topk, hidden_dim in configs:
            for with_hidden in (True, False):
                with self.subTest(
                    pool_size=pool_size,
                    m=m,
                    topk=topk,
                    hidden_dim=hidden_dim,
                    with_hidden=with_hidden,
                ):
                    bufs = _make_buffers(
                        pool_size,
                        topk,
                        hidden_dim,
                        with_hidden=with_hidden,
                        device=self.device,
                    )
                    indices = torch.randint(
                        0, pool_size, (m,), dtype=torch.int64, device=self.device
                    )
                    self._assert_matches_reference(indices, bufs)

    def test_empty_indices_returns_empty_rows(self):
        # m == 0 hits the early-return path; outputs must still carry the right
        # trailing dims / dtypes so downstream concatenation stays valid.
        for with_hidden in (True, False):
            with self.subTest(with_hidden=with_hidden):
                bufs = _make_buffers(
                    32, 4, 256, with_hidden=with_hidden, device=self.device
                )
                indices = torch.empty(0, dtype=torch.int64, device=self.device)
                self._assert_matches_reference(indices, bufs)

    def test_non_contiguous_indices(self):
        # indices flows from filtered/merged producers and can be strided; the
        # kernel addresses it linearly and relies on the internal .contiguous().
        pool_size, m = 256, 64
        bufs = _make_buffers(pool_size, 8, 512, with_hidden=True, device=self.device)
        pairs = torch.randint(
            0, pool_size, (m, 2), dtype=torch.int64, device=self.device
        )
        indices = pairs[:, 0]
        self.assertFalse(indices.is_contiguous(), "test setup: indices must be strided")
        self._assert_matches_reference(indices, bufs)

    def test_duplicate_indices(self):
        # Gather (not scatter): repeated source rows are well-defined and must
        # each produce an identical copy.
        pool_size, m = 8, 64
        bufs = _make_buffers(pool_size, 4, 333, with_hidden=True, device=self.device)
        indices = torch.randint(
            0, 3, (m,), dtype=torch.int64, device=self.device
        )  # tiny range -> many duplicates
        self._assert_matches_reference(indices, bufs)

    def test_index_dtype_variants(self):
        pool_size, m = 128, 50
        bufs = _make_buffers(pool_size, 8, 1024, with_hidden=True, device=self.device)
        base = torch.randint(0, pool_size, (m,), device=self.device)
        for idx_dtype in (torch.int32, torch.int64):
            with self.subTest(idx_dtype=idx_dtype):
                self._assert_matches_reference(base.to(idx_dtype), bufs)

    def test_hidden_dtype_variants(self):
        pool_size, m = 96, 40
        indices = torch.randint(
            0, pool_size, (m,), dtype=torch.int64, device=self.device
        )
        for hidden_dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(hidden_dtype=hidden_dtype):
                bufs = _make_buffers(
                    pool_size,
                    8,
                    2048,
                    with_hidden=True,
                    hidden_dtype=hidden_dtype,
                    device=self.device,
                )
                self._assert_matches_reference(indices, bufs)

    def test_outputs_do_not_alias_source_buffers(self):
        pool_size, m = 64, 32
        bufs = _make_buffers(pool_size, 8, 512, with_hidden=True, device=self.device)
        indices = torch.randint(
            0, pool_size, (m,), dtype=torch.int64, device=self.device
        )
        outputs = gather_spec_extras(indices, *bufs)
        for name, out, buf in zip(_OUTPUT_NAMES, outputs, bufs):
            if out is None or buf is None:
                continue
            self.assertNotEqual(
                out.data_ptr(),
                buf.data_ptr(),
                f"{name} output aliases its source buffer",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestScatterSpecExtras(CustomTestCase):
    """scatter_spec_extras must leave every relay buffer byte-identical to the
    advanced-index assignments FutureMap.stash runs without it."""

    def _sources(self, m, topk, hidden_dim, bonus_dtype, hidden_dtype, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        bonus = torch.randint(0, 2**31 - 1, (m,), device="cuda", generator=g).to(
            bonus_dtype
        )
        topk_p = torch.rand((m, topk), device="cuda", generator=g)
        topk_index = torch.randint(0, 2**40, (m, topk), device="cuda", generator=g)
        hidden = torch.randn((m, hidden_dim), device="cuda", generator=g).to(
            hidden_dtype
        )
        return bonus, topk_p, topk_index, hidden

    def _check(self, indices, sources, bufs, with_topk, with_hidden):
        bonus, topk_p, topk_index, hidden = sources
        ref = [None if b is None else b.clone() for b in bufs]
        ref[2][indices] = bonus.to(ref[2].dtype)
        if with_topk:
            ref[0][indices] = topk_p.to(ref[0].dtype)
            ref[1][indices] = topk_index.to(ref[1].dtype)
        if with_hidden:
            ref[3][indices] = hidden.to(ref[3].dtype)
        topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf = bufs
        self.assertTrue(
            scatter_spec_extras(
                indices,
                bonus,
                output_tokens_buf,
                topk_p if with_topk else None,
                topk_p_buf,
                topk_index if with_topk else None,
                topk_index_buf,
                hidden if with_hidden else None,
                hidden_states_buf,
            )
        )
        for r, b in zip(ref, bufs):
            if b is not None:
                self.assertTrue(torch.equal(r.view(torch.uint8), b.view(torch.uint8)))

    def test_matches_index_assignment(self):
        configs = [
            (16, 8, 1, 7),
            (64, 33, 4, 128),
            (128, 128, 8, 1024),
            (100, 8, 1, 4096),
            (2048, 777, 8, 5120),
        ]
        for (
            pool_size,
            m,
            topk,
            hidden_dim,
        ), with_topk, with_hidden, bonus_dtype in itertools.product(
            configs, (True, False), (True, False), (torch.int32, torch.int64)
        ):
            with self.subTest(
                pool_size=pool_size,
                m=m,
                topk=topk,
                hidden_dim=hidden_dim,
                with_topk=with_topk,
                with_hidden=with_hidden,
                bonus_dtype=bonus_dtype,
            ):
                bufs = _make_buffers(
                    pool_size, topk, hidden_dim, with_hidden=True, device="cuda"
                )
                sources = self._sources(
                    m, topk, hidden_dim, bonus_dtype, torch.bfloat16, pool_size
                )
                indices = torch.randperm(pool_size, device="cuda")[:m]
                self._check(indices, sources, list(bufs), with_topk, with_hidden)

    def test_index_layouts_and_hidden_dtypes(self):
        pool_size, m = 96, 40
        for idx_dtype, strided, hidden_dtype in itertools.product(
            (torch.int32, torch.int64),
            (False, True),
            (torch.bfloat16, torch.float16, torch.float32),
        ):
            with self.subTest(
                idx_dtype=idx_dtype, strided=strided, hidden_dtype=hidden_dtype
            ):
                bufs = _make_buffers(
                    pool_size,
                    8,
                    2048,
                    with_hidden=True,
                    hidden_dtype=hidden_dtype,
                    device="cuda",
                )
                sources = self._sources(m, 8, 2048, torch.int32, hidden_dtype, 1)
                perm = torch.randperm(pool_size, device="cuda").to(idx_dtype)
                indices = (
                    torch.stack([perm[:m], perm[m : 2 * m]], dim=1)[:, 0]
                    if strided
                    else perm[:m]
                )
                self._check(indices, sources, list(bufs), True, True)

    def test_refusals_leave_buffers_untouched(self):
        pool_size, m = 32, 8
        bufs = _make_buffers(pool_size, 4, 256, with_hidden=True, device="cuda")
        before = [b.clone() for b in bufs]
        bonus, topk_p, topk_index, hidden = self._sources(
            m, 4, 256, torch.int32, torch.bfloat16, 2
        )
        indices = torch.randperm(pool_size, device="cuda")[:m]
        topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf = bufs
        for args in (
            (bonus, topk_p.half(), topk_index, hidden),  # float cast
            (bonus, topk_p, topk_index, hidden.float()),  # float cast
            (bonus, topk_p, None, hidden),  # top-k pair incomplete
            (bonus, topk_p, topk_index, hidden.t().contiguous().t()),  # strided source
            (bonus[:-1], topk_p, topk_index, hidden),  # row count
        ):
            b, p, i, h = args
            self.assertFalse(
                scatter_spec_extras(
                    indices,
                    b,
                    output_tokens_buf,
                    p,
                    topk_p_buf,
                    i,
                    topk_index_buf,
                    h,
                    hidden_states_buf,
                )
            )
        for x, y in zip(before, bufs):
            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
