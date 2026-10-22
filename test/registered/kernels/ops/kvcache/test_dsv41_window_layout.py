"""Fused metadata agrees with the CPU reference, including graph refreshes."""

import unittest

import torch

from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestWindowLayout(unittest.TestCase):
    def compare(self, actual, expected):
        for name in expected.__struct_fields__:
            a, b = getattr(actual, name), getattr(expected, name)
            if isinstance(b, torch.Tensor):
                self.assertEqual(a.dtype, b.dtype, name)
                torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0, msg=name)
            else:
                self.assertEqual(a, b, name)

    def test_reference(self):
        for sizes in ([1], [3, 1, 260, 7], [4096], [4096] * 8):
            for window in (7, 128):
                for dtype in (torch.int32, torch.int64):
                    req = torch.cat(
                        [
                            torch.full((n,), [9, 2, 9, 5][i % 4], dtype=dtype)
                            for i, n in enumerate(sizes)
                        ]
                    )
                    pos = torch.cat(
                        [
                            torch.arange(n, dtype=dtype) + (0 if i % 2 else 123456)
                            for i, n in enumerate(sizes)
                        ]
                    )
                    floor = torch.cat(
                        [
                            torch.full((n,), 123400 if i % 2 == 0 else 0, dtype=dtype)
                            for i, n in enumerate(sizes)
                        ]
                    )
                    for use_floor in (False, True):
                        for padding in (None, len(req) + 13):
                            kw = dict(
                                zero_row=12345678,
                                window=window,
                                capacity=256,
                                num_groups=len(sizes) + 2,
                                padded_rows=padding,
                            )
                            cpu = window_layout(
                                req, pos, floor=floor if use_floor else None, **kw
                            )

                            # Noncontiguous input vectors exercise input strides.
                            def gpu(t):
                                return torch.stack([t, t], dim=1).cuda()[:, 0]

                            actual = window_layout(
                                gpu(req),
                                gpu(pos),
                                floor=gpu(floor) if use_floor else None,
                                **kw,
                            )
                            self.compare(actual, cpu)

    def test_graph_refresh(self):
        req = torch.tensor([1, 1, 1, 2, 2, 2], device="cuda")
        pos = torch.tensor([256, 257, 258, 0, 1, 2], device="cuda")
        floor = torch.zeros_like(pos)
        kw = dict(zero_row=99999, num_groups=6, padded_rows=10)
        for _ in range(3):
            window_layout(req, pos, floor=floor, **kw)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = window_layout(req, pos, floor=floor, **kw)
        for r, p, f in [
            ([5, 5, 8, 8, 8, 5], [0, 1, 512, 513, 514, 70], [0, 0, 510, 510, 510, 0]),
            ([3] * 6, list(range(200, 206)), [195] * 6),
        ]:
            req.copy_(torch.tensor(r, device="cuda"))
            pos.copy_(torch.tensor(p, device="cuda"))
            floor.copy_(torch.tensor(f, device="cuda"))
            graph.replay()
            self.compare(
                captured, window_layout(req.cpu(), pos.cpu(), floor=floor.cpu(), **kw)
            )


if __name__ == "__main__":
    unittest.main()
