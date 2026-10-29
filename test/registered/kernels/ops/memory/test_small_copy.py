import itertools
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.memory import small_field_copy as small_field_copy_module
from sglang.kernels.ops.memory.small_copy import (
    MAX_FIELD_NUMEL,
    _small_copy_kernel,
    try_small_copy,
)
from sglang.kernels.ops.memory.small_field_copy import (
    SmallFieldCopyPlan,
    small_field_copy,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _as_bytes(tensor):
    return (
        torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        .copy_(tensor)
        .view(torch.uint8)
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSmallCopy(CustomTestCase):
    def test_bits(self):
        for n, strided in itertools.product((0, 1, 8, 128, 4096), (False, True)):
            with self.subTest(n=n, strided=strided):
                dsts = []
                srcs = []
                backings = []
                for dtype in _DTYPES:
                    source = torch.randint(
                        0,
                        256,
                        (2 * n * torch.empty((), dtype=dtype).element_size(),),
                        device="cuda",
                        dtype=torch.uint8,
                    ).view(dtype)
                    src = source[::2] if strided else source[:n]
                    backing = torch.zeros(n * 3, device="cuda", dtype=dtype)
                    dst = backing[::3] if strided else backing[:n]
                    dsts.append(dst)
                    srcs.append(src)
                    backings.append(backing)
                expected = [s.clone() for s in srcs]
                self.assertTrue(try_small_copy(dsts, srcs))
                for d, s in zip(dsts, expected):
                    self.assertTrue(torch.equal(_as_bytes(d), _as_bytes(s)))

    def test_cast_and_mrope_graph(self):
        for n in (1, 4, 128, 2048):
            with self.subTest(n=n):
                s0 = torch.randint(
                    -(2**40), 2**40, (n * 2,), device="cuda", dtype=torch.int64
                )[::2]
                s1 = torch.arange(n * 3, device="cuda", dtype=torch.int32).reshape(3, n)
                d0 = torch.empty(n, device="cuda", dtype=torch.int32)
                backing = torch.full((3, n + 5), -7, device="cuda", dtype=torch.int64)
                d1 = backing[:, :n]
                try_small_copy([d0, d1], [s0, s1])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.assertTrue(try_small_copy([d0, d1], [s0, s1]))
                for _ in range(3):
                    s0.add_(17)
                    s1.add_(13)
                    graph.replay()
                    self.assertTrue(torch.equal(d0, s0.int()))
                    self.assertTrue(torch.equal(d1, s1.long()))
                    self.assertTrue(bool((backing[:, n:] == -7).all()))

    def test_fallbacks(self):
        x = torch.arange(17, device="cuda")
        y = torch.empty_like(x)
        self.assertFalse(try_small_copy([x[1:], y[:16]], [x[:-1], x[:16]]))
        self.assertFalse(try_small_copy([x, y], [y, x]))
        self.assertFalse(try_small_copy([x[:8], x[4:12]], [y[:8], y[:8]]))
        self.assertFalse(
            try_small_copy(
                [torch.empty(MAX_FIELD_NUMEL + 1, device="cuda") for _ in range(2)],
                [torch.empty(MAX_FIELD_NUMEL + 1, device="cuda") for _ in range(2)],
            )
        )
        self.assertFalse(
            try_small_copy(
                [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
                [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
            )
        )
        a = torch.empty(2)
        self.assertFalse(try_small_copy([a, a], [a, a]))
        self.assertTrue(try_small_copy([x, y], [x, y]))

    def test_hidden_rows_ride_with_small_fields(self):
        # a draft step's hidden rows and DSA seed next to its index fields
        for rows in (1, 2, 8):
            with self.subTest(rows=rows):
                hidden = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
                seed = torch.randint(
                    -1, 1 << 20, (rows, 2048), device="cuda", dtype=torch.int32
                )
                lens = torch.randint(1, 1 << 16, (rows,), device="cuda")
                dsts = [torch.empty_like(t) for t in (hidden, seed, lens)]
                self.assertTrue(try_small_copy(dsts, [hidden, seed, lens]))
                for d, src in zip(dsts, (hidden, seed, lens)):
                    self.assertTrue(torch.equal(_as_bytes(d), _as_bytes(src)))

    def test_broadcast_and_negative_views(self):
        source = torch.arange(7, device="cuda", dtype=torch.float32)
        expanded = source[None, :].expand(3, 7)
        destinations = [torch.empty_like(expanded), torch.empty_like(source)]
        self.assertTrue(try_small_copy(destinations, [expanded, source]))
        self.assertTrue(torch.equal(destinations[0], expanded))
        self.assertTrue(torch.equal(destinations[1], source))
        self.assertFalse(try_small_copy(destinations, [expanded, source._neg_view()]))

    def test_grouped_copy_fast_path_matches_the_copies(self):
        """With the fill fast path, one dense integer dtype pair takes one foreach
        launch; every group shape still writes exactly the copies."""
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            _grouped_foreach_copy_,
        )

        cuda = dict(device="cuda")
        backing = torch.full((64,), -1, dtype=torch.int64, **cuda)
        mrope = torch.full((3, 16), -1, dtype=torch.int64, **cuda)
        mask = torch.zeros(4, dtype=torch.bool, **cuda)
        cases = (
            (
                [backing[:4], backing[8:13]],
                [torch.arange(4, **cuda), torch.arange(5, **cuda) + 9],
            ),
            ([backing[16:20]], [torch.arange(4, dtype=torch.int32, **cuda)]),
            (
                [backing[24:28], mask],
                [torch.arange(4, **cuda), torch.ones(4, dtype=torch.bool, **cuda)],
            ),
            (
                [backing[32:36], mrope[:, :4]],
                [torch.arange(4, **cuda), torch.arange(12, **cuda).view(3, 4)],
            ),
        )
        with envs.SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH.override(True):
            for dsts, srcs in cases:
                _grouped_foreach_copy_(dsts, srcs)
                for dst, src in zip(dsts, srcs):
                    self.assertTrue(torch.equal(dst, src.to(dst.dtype)))
        self.assertTrue(bool((backing[13:16] == -1).all()))
        self.assertTrue(bool((mrope[:, 4:] == -1).all()))

    def test_dynamic_metadata_has_bounded_specializations(self):
        kernel_cache = _small_copy_kernel.device_caches[torch.cuda.current_device()][0]
        initial_count = len(kernel_cache)
        specialization_count = None
        for n in [*range(1, 130), 255, 256, 511, 512, 1023, 1024, 2048, 2730]:
            step = 1 + n % 3
            source = torch.arange(n * step, device="cuda", dtype=torch.int64)[::step]
            backing = torch.full((n * step,), -7, device="cuda", dtype=torch.int32)
            destination = backing[::step]
            matrix = torch.arange(
                3 * (n + 3), device="cuda", dtype=torch.int32
            ).reshape(3, n + 3)[:, :n]
            matrix_backing = torch.full(
                (3, n + 5), -7, device="cuda", dtype=torch.int64
            )
            matrix_destination = matrix_backing[:, :n]
            self.assertTrue(
                try_small_copy([destination, matrix_destination], [source, matrix])
            )
            self.assertTrue(torch.equal(destination, source.int()))
            self.assertTrue(torch.equal(matrix_destination, matrix.long()))
            self.assertTrue(bool((matrix_backing[:, n:] == -7).all()))
            for offset in range(1, step):
                self.assertTrue(bool((backing[offset::step] == -7).all()))
            if n == 48:
                specialization_count = len(kernel_cache)
                self.assertLessEqual(specialization_count - initial_count, 16)
            elif n > 48:
                self.assertEqual(len(kernel_cache), specialization_count)


def _raw_bits(tensor):
    # Same-width integer view: compares stored bits without normalizing bool or NaN payloads.
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return tensor.view(width[tensor.element_size()])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSmallFieldCopy(CustomTestCase):
    """small_field_copy must match try_small_copy: same bytes, same refusals."""

    def test_bits_match_reference_copy(self):
        for n, strided in itertools.product((0, 1, 8, 128, 4096), (False, True)):
            with self.subTest(n=n, strided=strided):
                dsts, srcs, refs = [], [], []
                for dtype in _DTYPES:
                    source = torch.randint(
                        0,
                        256,
                        (2 * n * torch.empty((), dtype=dtype).element_size(),),
                        device="cuda",
                        dtype=torch.uint8,
                    ).view(dtype)
                    src = source[::2] if strided else source[:n]
                    dst = torch.zeros(n * 3, device="cuda", dtype=dtype)
                    dsts.append(dst[::3] if strided else dst[:n])
                    srcs.append(src)
                    refs.append(torch.zeros_like(dsts[-1]))
                for chunk in range(0, len(dsts), 5):
                    self.assertTrue(
                        small_field_copy(
                            dsts[chunk : chunk + 5], srcs[chunk : chunk + 5]
                        )
                    )
                    self.assertTrue(
                        try_small_copy(refs[chunk : chunk + 5], srcs[chunk : chunk + 5])
                    )
                for d, r in zip(dsts, refs):
                    self.assertTrue(torch.equal(_raw_bits(d), _raw_bits(r)))

    def test_width_casts_rows_and_graph_replay(self):
        for n in (1, 4, 128, 2048):
            with self.subTest(n=n):
                s0 = torch.randint(
                    -(2**40), 2**40, (n * 2,), device="cuda", dtype=torch.int64
                )[::2]
                s1 = torch.arange(n * 3, device="cuda", dtype=torch.int32).reshape(3, n)
                d0 = torch.empty(n, device="cuda", dtype=torch.int32)
                backing = torch.full((3, n + 5), -7, device="cuda", dtype=torch.int64)
                d1 = backing[:, :n]
                self.assertTrue(small_field_copy([d0, d1], [s0, s1]))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.assertTrue(small_field_copy([d0, d1], [s0, s1]))
                for _ in range(3):
                    s0.add_(17)
                    s1.add_(13)
                    graph.replay()
                    self.assertTrue(torch.equal(d0, s0.int()))
                    self.assertTrue(torch.equal(d1, s1.long()))
                    self.assertTrue(bool((backing[:, n:] == -7).all()))

    def test_refusals_leave_destinations_untouched(self):
        x = torch.arange(17, device="cuda")
        y = torch.full_like(x, -1)
        self.assertFalse(small_field_copy([x[1:], y[:16]], [x[:-1], x[:16]]))
        self.assertFalse(small_field_copy([x, y], [y, x]))
        self.assertFalse(small_field_copy([x[:8], x[4:12]], [y[:8], y[:8]]))
        self.assertTrue(torch.equal(x, torch.arange(17, device="cuda")))
        self.assertTrue(bool((y == -1).all()))
        self.assertFalse(
            small_field_copy(
                [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
                [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
            )
        )
        a = torch.empty(2)
        self.assertFalse(small_field_copy([a, a], [a, a]))
        self.assertTrue(small_field_copy([x, y], [x, y]))

    def test_plan_refreshes_sources_and_revalidates(self):
        """A plan copies from each call's sources, re-validates a source whose layout
        changed, and still refuses overlaps and foreign devices without writing."""
        plan = SmallFieldCopyPlan()
        d0 = torch.empty(8, device="cuda", dtype=torch.int64)
        d1 = torch.empty((2, 4), device="cuda", dtype=torch.int32)
        for step in range(4):
            s0 = torch.randint(
                -(2**31), 2**31 - 1, (8,), device="cuda", dtype=torch.int32
            )
            s1 = torch.randint(0, 100, (2, 4), device="cuda", dtype=torch.int32)
            if step == 2:
                s1 = torch.randint(0, 100, (4, 4), device="cuda", dtype=torch.int32)[
                    ::2
                ]
            self.assertTrue(plan([d0, d1], [s0, s1]))
            self.assertTrue(torch.equal(d0, s0.long()))
            self.assertTrue(torch.equal(d1, s1))
        before0, before1 = d0.clone(), d1.clone()
        self.assertFalse(plan([d0, d1], [d1.view(-1), s1]))
        self.assertFalse(plan([d0, d1], [s0.cpu(), s1]))
        self.assertTrue(torch.equal(d0, before0) and torch.equal(d1, before1))
        self.assertTrue(plan([d0, d1], [s0, s1]))
        self.assertTrue(torch.equal(d0, s0.long()))

    def test_plan_reads_pinned_host_sources_only_when_allowed(self):
        """The post-verify graphs stage a host-drawn scalar with their device inputs
        in one launch: a plan that allows it reads page-locked host memory, and every
        other host source is refused without a write."""
        host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        src = torch.arange(8, device="cuda")
        d0 = torch.empty(8, device="cuda", dtype=torch.int64)
        d1 = torch.full((1,), -1, device="cuda", dtype=torch.int64)
        plan = SmallFieldCopyPlan(allow_pinned_src=True)
        for value in (3, 5):
            host[0] = value
            self.assertTrue(plan([d0, d1], [src, host]))
            torch.cuda.synchronize()
            self.assertEqual((int(d1.item()), d0.tolist()), (value, src.tolist()))
        d1.fill_(-1)
        self.assertFalse(SmallFieldCopyPlan()([d0, d1], [src, host]))
        pageable = torch.ones(1, dtype=torch.int64)
        self.assertFalse(
            SmallFieldCopyPlan(allow_pinned_src=True)([d0, d1], [src, pageable])
        )
        self.assertEqual(int(d1.item()), -1)

    def test_vec_launch_matches_default_bytes(self):
        """The vectorized launch writes the same bytes as the default one, including
        misaligned, strided and width-cast fields, and leaves row padding untouched."""
        gen = torch.Generator(device="cuda").manual_seed(0)

        def field(dtype, rows, cols, src_pad, dst_pad, offset, cast):
            size = torch.empty((), dtype=dtype).element_size()
            numel = offset + rows * (cols + src_pad)
            src = torch.randint(
                0, 256, (numel * size,), device="cuda", dtype=torch.uint8, generator=gen
            ).view(dtype)[offset:]
            src = src.view(rows, cols + src_pad)[:, :cols]
            dst_dtype = (
                {torch.int32: torch.int64, torch.int64: torch.int32}.get(dtype)
                if cast
                else dtype
            )
            backing = torch.randint(
                0,
                256,
                ((rows * (cols + dst_pad) + offset) * 8,),
                device="cuda",
                dtype=torch.uint8,
                generator=gen,
            )
            return src, backing, dst_dtype, offset

        def views(backing, dst_dtype, offset, rows, cols, dst_pad):
            numel = rows * (cols + dst_pad) + offset
            flat = backing[
                : numel * torch.empty((), dtype=dst_dtype).element_size()
            ].view(dst_dtype)
            dst = flat[offset:].view(rows, cols + dst_pad)[:, :cols]
            return dst if rows > 1 else dst.reshape(cols)

        cases = [
            # (dtype, rows, cols, src_pad, dst_pad, offset, cast)
            [
                (torch.bfloat16, 56, 4096, 0, 0, 0, False),
                (torch.int64, 1, 8, 0, 0, 0, False),
                (torch.int32, 1, 8, 0, 0, 0, True),
                (torch.int32, 1, 9, 0, 0, 0, False),
                (torch.int64, 1, 56, 0, 0, 0, False),
                (torch.int64, 3, 8, 0, 0, 0, False),
            ],
            [
                (torch.bfloat16, 8, 4096, 0, 0, 1, False),
                (torch.float32, 8, 1, 0, 0, 0, False),
                (torch.int64, 1, 64, 0, 0, 1, True),
                (torch.uint8, 5, 33, 3, 7, 0, False),
            ],
            [
                (torch.bfloat16, 7, 4096, 8, 16, 0, False),
                (torch.float16, 4, 24, 8, 8, 0, False),
                (torch.float64, 3, 2, 2, 0, 0, False),
                (torch.int16, 2, 16, 0, 8, 0, False),
                (torch.bool, 1, 100, 0, 0, 0, False),
                (torch.int32, 2, 6, 2, 2, 0, True),
            ],
            [
                (torch.bfloat16, 512, 4096, 0, 0, 0, False),
                (torch.int8, 1, 1, 0, 0, 0, False),
            ],
        ]
        for case in cases:
            with self.subTest(case=case):
                built = [(spec, field(*spec)) for spec in case]
                outs = []
                for vec in (False, True):
                    dsts, srcs, backings = [], [], []
                    for (dtype, rows, cols, _, dst_pad, _, _), (
                        src,
                        backing,
                        dst_dtype,
                        offset,
                    ) in built:
                        b = backing.clone()
                        dsts.append(views(b, dst_dtype, offset, rows, cols, dst_pad))
                        srcs.append(src if rows > 1 else src.reshape(cols))
                        backings.append(b)
                    with mock.patch.object(small_field_copy_module, "_VEC", vec):
                        self.assertTrue(small_field_copy(dsts, srcs))
                        plan = SmallFieldCopyPlan()
                        self.assertTrue(plan(dsts, srcs))
                        self.assertTrue(plan(dsts, srcs))
                    outs.append(backings)
                for ref, got in zip(*outs):
                    self.assertTrue(torch.equal(ref, got))

    def test_vec_launch_refusals_and_graph_replay(self):
        with mock.patch.object(small_field_copy_module, "_VEC", True):
            x = torch.arange(17, device="cuda")
            y = torch.full_like(x, -1)
            self.assertFalse(small_field_copy([x[1:], y[:16]], [x[:-1], x[:16]]))
            self.assertFalse(small_field_copy([x, y], [y, x]))
            self.assertTrue(torch.equal(x, torch.arange(17, device="cuda")))
            self.assertTrue(bool((y == -1).all()))
            src = torch.randn(56, 4096, device="cuda").bfloat16()
            idx = torch.arange(8, device="cuda", dtype=torch.int32)
            dst = torch.empty_like(src)
            dst_idx = torch.empty(8, device="cuda", dtype=torch.int64)
            plan = SmallFieldCopyPlan()
            self.assertTrue(plan([dst, dst_idx], [src, idx]))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.assertTrue(plan([dst, dst_idx], [src, idx]))
            for _ in range(3):
                src.add_(1)
                idx.add_(3)
                graph.replay()
                self.assertTrue(
                    torch.equal(dst.view(torch.int16), src.view(torch.int16))
                )
                self.assertTrue(torch.equal(dst_idx, idx.long()))


if __name__ == "__main__":
    unittest.main()
