"""Request-window copies preserve data, scales, padding and graph addresses."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.dsv41_window_layout import _window_layout_queries
from sglang.srt.mem_cache.dsv41_request_window import (
    RequestWindowSpec,
    copy_packed_tokens,
    window_layout,
)
from sglang.srt.mem_cache.kv.memory.request_window import RequestWindowMemory
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def reference(src, dst, source, target, page, layout):
    # Independent scalar row slicing also checks every untouched padding byte.
    result = dst.clone()
    for s, d in zip(source.tolist(), target.tolist()):
        if layout.is_interleaved:
            rows = ((layout.bytes_per_token, 0),)
        else:
            rows = (
                (layout.data_bytes, 0),
                (layout.scale_bytes, page * layout.data_bytes),
            )
        for width, base in rows:
            result[
                d // page, base + d % page * width : base + (d % page + 1) * width
            ] = src[s // page, base + s % page * width : base + (s % page + 1) * width]
    return result


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRequestWindowCopy(CustomTestCase):
    def test_layouts_pages_and_strides(self):
        for layout in KVLayout:
            for page in (16, 64, 256):
                for count in (0, 1, 17, 129):
                    with self.subTest(layout=layout, page=page, count=count):
                        width = layout.page_bytes(page)
                        src = torch.randint(
                            0, 256, (12, width + 64), dtype=torch.uint8
                        )[:, :width]
                        dst = torch.full((12, width + 128), 193, dtype=torch.uint8)[
                            :, :width
                        ]
                        source = (torch.arange(count) * 7 + page - 1) % (12 * page)
                        target = torch.randperm(12 * page)[:count]
                        expected = reference(src, dst, source, target, page, layout)
                        sg = torch.empty(
                            (12, width + 64), dtype=torch.uint8, device="cuda"
                        )[:, :width]
                        dg = torch.empty(
                            (12, width + 128), dtype=torch.uint8, device="cuda"
                        )[:, :width]
                        sg.copy_(src)
                        dg.copy_(dst)
                        # Non-contiguous location arrays, mixed integer widths.
                        sl = torch.zeros(count * 2, dtype=torch.int32, device="cuda")[
                            ::2
                        ]
                        dl = torch.zeros(count * 2, dtype=torch.int64, device="cuda")[
                            ::2
                        ]
                        sl.copy_(source)
                        dl.copy_(target)
                        copy_packed_tokens(
                            sg, dg, sl, dl, page_size=page, layout=layout
                        )
                        torch.testing.assert_close(dg.cpu(), expected, rtol=0, atol=0)

    def test_window_layout_dynamic_bounds_reuse_query_variant(self):
        # Count query JIT variants, not history kernels or changes to the
        # legitimate constexpr layout (strides and floor presence).
        def compiled_variants():
            caches = getattr(_window_layout_queries, "device_caches", None)
            if caches is None:
                return None
            return sum(
                len(v[0]) if isinstance(v, tuple) else len(v) for v in caches.values()
            )

        cases = (
            ((1,), 1, 1),
            ((16,), 1, 16),
            ((9, 8), 2, 33),
            ((1,) * 16, 16, 32),
            ((2,) * 17, 20, 35),
            ((300, 17, 1), 4, 320),
        )

        def cuda_strided(cpu, stride):
            gpu = torch.empty(cpu.numel() * stride, dtype=cpu.dtype, device="cuda")[
                ::stride
            ]
            gpu.copy_(cpu)
            return gpu

        for has_floor, strides in ((False, (1, 1, 1)), (True, (2, 3, 4))):
            variants = None
            for lens, groups, padded in cases:
                with self.subTest(
                    lens=lens, groups=groups, padded=padded, floor=has_floor
                ):
                    prefixes = [0, 147, 8191, (1 << 20) - 1024]
                    starts = [prefixes[i % len(prefixes)] for i in range(len(lens))]
                    req = torch.cat(
                        [torch.full((n,), i + 1) for i, n in enumerate(lens)]
                    )
                    pos = torch.cat(
                        [torch.arange(s, s + n) for s, n in zip(starts, lens)]
                    )
                    floor = (
                        torch.cat(
                            [
                                torch.full((n,), max(0, s - 32))
                                for s, n in zip(starts, lens)
                            ]
                        )
                        if has_floor
                        else None
                    )
                    kwargs = dict(zero_row=32768, num_groups=groups, padded_rows=padded)
                    expected = window_layout(req, pos, floor=floor, **kwargs)
                    actual = window_layout(
                        cuda_strided(req, strides[0]),
                        cuda_strided(pos, strides[1]),
                        floor=cuda_strided(floor, strides[2]) if has_floor else None,
                        **kwargs,
                    )
                    self.assertEqual(actual.size, expected.size)
                    for field in expected.__struct_fields__:
                        if field != "size":
                            torch.testing.assert_close(
                                getattr(actual, field).cpu(),
                                getattr(expected, field),
                                rtol=0,
                                atol=0,
                            )
                    count = compiled_variants()
                    if variants is not None:
                        self.assertEqual(count, variants)
                    variants = count

    def test_graph_replay_refreshes_locations(self):
        page, layout = 64, KVLayout.V41
        src = torch.randint(
            0, 256, (8, layout.page_bytes(page)), dtype=torch.uint8, device="cuda"
        )
        dst = torch.full_like(src, 193)
        source = torch.arange(128, dtype=torch.int64, device="cuda")
        target = source.clone()
        copy_packed_tokens(src, dst, source, target, page_size=page, layout=layout)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy_packed_tokens(src, dst, source, target, page_size=page, layout=layout)
        for offset in (63, 127, 256):
            source.copy_(torch.arange(128, device="cuda") + offset)
            target.copy_(torch.arange(128, device="cuda") + 128)
            dst.fill_(193)
            expected = reference(
                src.cpu(), dst.cpu(), source.cpu(), target.cpu(), page, layout
            )
            graph.replay()
            torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)

    def test_commit_retained_rows(self):
        for fmt in KVLayout:
            for lens in ([1, 17, 257], [16384], [300, 37, 600]):
                with self.subTest(layout=fmt, lengths=lens):
                    self._check_commit(fmt, lens, graph=False)

    def test_commit_graph_refreshes_metadata(self):
        for fmt in KVLayout:
            self._check_commit(fmt, [300, 37, 600], graph=True)

    def _check_commit(self, fmt, lens, graph):
        page, capacity = 64, 256

        spec = RequestWindowSpec.create(
            num_slots=8,
            layers=2,
            page_size=page,
            capacity=capacity,
            workspace_rows=sum(lens) + len(lens) * 128,
            kv_layout=fmt,
        )
        memory = RequestWindowMemory(spec, dtype=torch.uint8, device="cuda")
        window = memory.view

        def metadata(lengths, shift):
            req = torch.cat(
                [
                    torch.full((n,), i + 1, dtype=torch.int64, device="cuda")
                    for i, n in enumerate(lengths)
                ]
            )
            pos = torch.cat(
                [torch.arange(n, device="cuda") + 191 + shift for n in lengths]
            )
            return window_layout(
                req,
                pos,
                zero_row=window.zero_row,
                capacity=capacity,
                num_groups=len(lengths),
            )

        active = metadata(lens, 0)
        window.activate(active)
        src = window.workspace.kv_buffer[0]
        src.random_(0, 256)
        # History preparation is already complete: test the real commit method.
        window.prepared = (1, False)
        window.commit(1)
        captured = None
        if graph:
            captured = torch.cuda.CUDAGraph()
            window.prepared = (1, True)
            with torch.cuda.graph(captured):
                window.commit(1)
        for iteration in range(3):
            lengths = [17, 320, 600] if graph and iteration % 2 else lens
            active.copy_(metadata(lengths, iteration * 137))
            src.random_(0, 256)
            window.state.kv_buffer[1].fill_(193)
            window.tags.fill_(-1)
            keep = active.commit_mask.cpu()
            source = active.write_loc.cpu()[keep]
            positions = active.pos.cpu()[keep]
            target = active.req.cpu()[keep] * capacity + positions % capacity
            expected = reference(
                src.cpu(), window.state.kv_buffer[1].cpu(), source, target, page, fmt
            )
            expected_tags = window.tags[1].cpu().clone()
            expected_tags[target] = positions
            if captured is None:
                window.prepared = (1, False)
                window.commit(1)
            else:
                captured.replay()
            torch.testing.assert_close(
                window.state.kv_buffer[1].cpu(), expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                window.tags[1].cpu(), expected_tags, rtol=0, atol=0
            )
            self.assertTrue(torch.all(window.tags[0] == -1).item())


if __name__ == "__main__":
    unittest.main()
