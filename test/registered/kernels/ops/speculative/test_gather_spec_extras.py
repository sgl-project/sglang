from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

import unittest

import torch

from sglang.kernels.ops.speculative.gather_spec_extras import (
    gather_spec_extras,
    scatter_spec_extras,
)
from sglang.test.test_utils import CustomTestCase

_OUTPUT_NAMES = (
    "topk_p",
    "topk_index",
    "bonus_tokens",
    "hidden_states",
    "dsa_topk_indices",
)


def _ref_gather(
    indices,
    topk_p_buf,
    topk_index_buf,
    output_tokens_buf,
    hidden_states_buf,
    dsa_topk_indices_buf=None,
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
    dsa_topk_indices = (
        dsa_topk_indices_buf[indices] if dsa_topk_indices_buf is not None else None
    )
    return topk_p, topk_index, bonus_tokens, hidden_states, dsa_topk_indices


def _make_buffers(
    pool_size,
    topk,
    hidden_dim,
    *,
    with_hidden,
    hidden_dtype=torch.bfloat16,
    device="cuda",
    seed=0,
    dsa_width=None,
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
    # the DSA top-k seed relayed with the hidden states (absent with them)
    dsa_width = dsa_width or (hidden_dim % 300 + 1)
    dsa_topk_indices_buf = (
        torch.randint(
            -1,
            1 << 20,
            (pool_size, dsa_width),
            dtype=torch.int32,
            device=device,
            generator=g,
        )
        if with_hidden
        else None
    )
    return (
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
        dsa_topk_indices_buf,
    )


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

    def test_seed_without_hidden_states(self):
        pool_size, m = 64, 20
        bufs = list(
            _make_buffers(pool_size, 4, 256, with_hidden=True, device=self.device)
        )
        bufs[3] = None
        indices = torch.randint(
            0, pool_size, (m,), dtype=torch.int64, device=self.device
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


def _ref_scatter(indices, pairs):
    """The index_put stash the fused scatter replaced: buf[indices] = rows.to(dtype)."""
    for p in pairs:
        if p is not None:
            rows, buf = p
            buf[indices] = rows.to(buf.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestScatterSpecExtras(CustomTestCase):
    def _pairs(self, pool, m, hidden, g, dev):
        def rows_and_buf(shape, rows_dtype, buf_dtype):
            if rows_dtype.is_floating_point:
                rows = torch.randn((m, *shape), generator=g, device=dev).to(rows_dtype)
                buf = torch.randn((pool, *shape), generator=g, device=dev).to(buf_dtype)
            else:
                rows = torch.randint(
                    -1, 1 << 20, (m, *shape), generator=g, device=dev
                ).to(rows_dtype)
                buf = torch.randint(
                    -1, 1 << 20, (pool, *shape), generator=g, device=dev
                ).to(buf_dtype)
            return rows, buf

        return [
            rows_and_buf((), torch.int32, torch.int64),  # bonus tokens
            rows_and_buf((1,), torch.float32, torch.float32),  # topk_p
            rows_and_buf((1,), torch.int64, torch.int64),  # topk_index
            rows_and_buf((hidden,), torch.bfloat16, torch.bfloat16),  # hidden
            rows_and_buf((3, 1000), torch.float32, torch.float32),  # draft probs
            rows_and_buf((2048,), torch.int32, torch.int32),  # DSA top-k seed
        ]

    def test_matches_index_put(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(0)
        for pool, m, hidden in ((16, 1, 6144), (64, 4, 6144), (300, 37, 4097)):
            for skip in (None, 0, 4, 5):
                with self.subTest(pool=pool, m=m, hidden=hidden, skip=skip):
                    pairs = self._pairs(pool, m, hidden, g, dev)
                    if skip is not None:
                        pairs[skip] = None
                    indices = torch.randperm(pool, generator=g, device=dev)[:m]
                    want = [None if p is None else p[1].clone() for p in pairs]
                    _ref_scatter(
                        indices,
                        [None if p is None else (p[0], w) for p, w in zip(pairs, want)],
                    )
                    scatter_spec_extras(indices, pairs)
                    for p, w in zip(pairs, want):
                        if p is not None:
                            torch.testing.assert_close(p[1], w, rtol=0, atol=0)

    def test_bonus_tokens_alone(self):
        # the relay of a non-speculative overlap decode step: one int32 column into
        # the int64 token buffer
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(1)
        for m in (1, 2, 3, 16, 129):
            with self.subTest(m=m):
                buf = torch.randint(0, 1 << 20, (512,), generator=g, device=dev)
                rows = torch.randint(
                    0, 1 << 20, (m,), generator=g, device=dev, dtype=torch.int32
                )
                indices = torch.randperm(512, generator=g, device=dev)[:m]
                want = buf.clone()
                _ref_scatter(indices, [(rows, want)])
                scatter_spec_extras(indices, [(rows, buf)])
                torch.testing.assert_close(buf, want, rtol=0, atol=0)

    def test_float_rows_cast_to_the_buffer_dtype(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(2)
        buf = torch.randn((64, 6144), generator=g, device=dev).to(torch.bfloat16)
        rows = torch.randn((5, 6144), generator=g, device=dev, dtype=torch.float32)
        indices = torch.randperm(64, generator=g, device=dev)[:5]
        want = buf.clone()
        want[indices] = rows.to(torch.bfloat16)
        scatter_spec_extras(indices, [(rows, buf)])
        torch.testing.assert_close(buf, want, rtol=0, atol=0)

    def test_non_contiguous_rows(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(3)
        buf = torch.zeros((32, 8), device=dev)
        rows = torch.randn((8, 6), generator=g, device=dev).t()  # (6, 8), strided
        self.assertFalse(rows.is_contiguous(), "test setup: rows must be strided")
        indices = torch.randperm(32, generator=g, device=dev)[:6]
        want = buf.clone()
        want[indices] = rows
        scatter_spec_extras(indices, [(rows, buf)])
        torch.testing.assert_close(buf, want, rtol=0, atol=0)

    def test_fewer_rows_than_indices(self):
        dev = torch.device("cuda")
        buf = torch.zeros((8, 4), dtype=torch.int32, device=dev)
        with self.assertRaises(ValueError):
            scatter_spec_extras(
                torch.arange(3, device=dev),
                [(torch.ones((2, 4), dtype=torch.int32, device=dev), buf)],
            )

    def test_empty_and_width_mismatch(self):
        dev = torch.device("cuda")
        buf = torch.zeros((8, 4), dtype=torch.int32, device=dev)
        scatter_spec_extras(
            torch.empty(0, dtype=torch.int64, device=dev),
            [(torch.empty((0, 4), dtype=torch.int32, device=dev), buf)],
        )
        self.assertEqual(int(buf.abs().sum()), 0)
        with self.assertRaises(ValueError):
            scatter_spec_extras(
                torch.arange(2, device=dev),
                [(torch.ones((2, 3), dtype=torch.int32, device=dev), buf)],
            )


class TestScatterSpecExtrasOffGpu(CustomTestCase):
    """Off the GPU the stash keeps one index_put per buffer."""

    def test_matches_index_put_on_cpu(self):
        g = torch.Generator().manual_seed(0)
        indices = torch.randperm(16, generator=g)[:5]
        pairs = [
            (
                torch.randint(0, 1000, (5,), dtype=torch.int32),
                torch.zeros(16, dtype=torch.int64),
            ),
            (torch.rand((5, 2)), torch.zeros((16, 2))),
            None,
            (torch.randn((5, 8)), torch.zeros((16, 8), dtype=torch.bfloat16)),
        ]
        want = [None if p is None else p[1].clone() for p in pairs]
        _ref_scatter(
            indices, [None if p is None else (p[0], w) for p, w in zip(pairs, want)]
        )
        scatter_spec_extras(indices, pairs)
        for p, w in zip(pairs, want):
            if p is not None:
                torch.testing.assert_close(p[1], w, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
