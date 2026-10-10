"""DCP unpack must preserve ragged token order and leave suffix slots untouched."""

import unittest

import torch

from sglang.kernels.ops.kvcache.dcp_gather import unpack_dcp_kv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDcpGather(CustomTestCase):
    def test_final_layouts(self):
        # Protect rank/token mapping, unaligned chunk starts, empty requests,
        # strided destinations, FP8 transport, and prefix/suffix boundaries.
        for world in (1, 2, 4, 8):
            for dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                for split in (False, True):
                    with self.subTest(world=world, dtype=dtype, split=split):
                        self._check_layout(world, dtype, split)

    def test_long_prefix_grid(self):
        # Token tiles must use grid.x: grid.y is limited to 65535 CTAs.
        self._check_layout(8, torch.bfloat16, True, long_prefix=True)

    def _check_layout(self, world, dtype, split, long_prefix=False):
        lengths, starts, suffixes = [19, 0, 4099, 1], [3, 0, 17, 2], [2, 3, 1, 0]
        if long_prefix:
            lengths, starts, suffixes = [262145], [3], [2]
        k_dim, pe_dim = 512, 64
        dim = k_dim + pe_dim
        rank_parts = [[] for _ in range(world)]
        metadata, expected = [], []
        padded_start = output_start = 0
        generator = torch.Generator().manual_seed(17)
        for n, start, suffix in zip(lengths, starts, suffixes):
            padded = ((start % world + n + world - 1) // world) * world
            rows = torch.randn(padded, 1, dim, generator=generator).to(dtype)
            # Striding a logical token sequence is the independent shard oracle.
            for rank in range(world):
                rank_parts[rank].append(rows.float()[rank::world])
            metadata.append([padded_start, start % world, n, output_start])
            expected.extend(
                (
                    rows.float()[start % world : start % world + n],
                    torch.full((suffix, 1, dim), -7.0),
                )
            )
            padded_start += padded // world
            output_start += n + suffix
        gathered = torch.cat([torch.cat(parts) for parts in rank_parts]).to(
            device="cuda", dtype=dtype
        )
        metadata = torch.tensor(
            metadata, dtype=torch.int64, device="cpu" if long_prefix else "cuda"
        )
        expected = torch.cat(expected).cuda()
        output_dtype = torch.bfloat16 if split else dtype
        if split:
            outputs = (
                torch.full(
                    (output_start, 1, k_dim), -7, device="cuda", dtype=output_dtype
                ),
                torch.full(
                    (output_start, 1, pe_dim), -7, device="cuda", dtype=output_dtype
                ),
            )
        else:
            combined = torch.full(
                (output_start, 1, dim), -7, device="cuda", dtype=output_dtype
            )
            outputs = combined.split([k_dim, pe_dim], dim=-1)
        unpack_dcp_kv(gathered, metadata, *outputs, world, max(lengths))
        actual = torch.cat([part.float() for part in outputs], dim=-1)
        torch.testing.assert_close(
            actual, expected.to(output_dtype).float(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
