"""Byte-exact pageable H2D, including bounded staging and packed draft layers."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.mem_cache.pool_host.staged_direct import StagedDirectLoad


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestStagedDirectLoad(unittest.TestCase):
    def test_fragmented_pages_layers_and_reused_staging(self):
        for page, row, dtype in (
            (64, 656, torch.uint8),
            (1, 8448, torch.uint8),
            (64, 576, torch.bfloat16),
            (64, 576, torch.float8_e4m3fn),
        ):
            with self.subTest(page=page, row=row, dtype=dtype):
                layers = 3  # last layer represents the packed draft tail
                host = torch.randint(0, 16, (12, layers, page, 1, row)).to(dtype)
                loader = StagedDirectLoad(
                    host, page, "cuda", max_bytes=2 * page * row * host.element_size()
                )
                dst = torch.zeros((12 * page, 1, row), dtype=dtype, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                for source, destination in (
                    ([9, 2, 7, 1, 5], [2, 8, 1, 5, 3]),
                    ([4, 8, 0], [11, 3, 7]),
                ):
                    hi = (
                        torch.tensor(source)[:, None] * page + torch.arange(page)
                    ).flatten()
                    di = (
                        torch.tensor(destination)[:, None] * page + torch.arange(page)
                    ).flatten()
                    for layer in range(layers):
                        with torch.cuda.stream(stream):
                            loader.load(dst, hi, di, layer)
                        stream.synchronize()
                        actual = dst.cpu().view(torch.uint8).index_select(0, di)
                        expected = (
                            host.view(torch.uint8)[:, layer]
                            .index_select(0, torch.tensor(source))
                            .reshape_as(actual)
                        )
                        self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(loader.cpu.is_pinned())
                before = dst.clone()
                with self.assertRaises(ValueError):
                    loader.load(dst, torch.arange(page + 1), torch.arange(page), 0)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(before.view(torch.uint8), dst.view(torch.uint8))
                )


if __name__ == "__main__":
    unittest.main()
