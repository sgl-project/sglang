import unittest
from unittest.mock import patch

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# gfx950 LDS per CU.
GFX950_LDS_BYTES = 160 * 1024


class TestMatmulPersistentConfigs(CustomTestCase):
    def test_gfx950_bf16_preset(self):
        """The gfx950 bf16 preset is a measured choice; a change must be deliberate
        and still fit gfx950's LDS."""
        with patch.object(batch_invariant_ops, "_is_gfx95", True):
            cfg = batch_invariant_ops._matmul_persistent_configs()[torch.bfloat16]
        self.assertEqual(
            cfg,
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
                "matrix_instr_nonkdim": 16,
            },
        )
        tile_bytes = (
            cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_K"]
            + cfg["BLOCK_SIZE_K"] * cfg["BLOCK_SIZE_N"]
        ) * torch.bfloat16.itemsize
        self.assertLessEqual(cfg["num_stages"] * tile_bytes, GFX950_LDS_BYTES)

    def test_other_gpus_keep_the_default(self):
        """The preset needs more LDS than gfx942 has, so it must stay gfx950-only."""
        with patch.object(batch_invariant_ops, "_is_gfx95", False):
            configs = batch_invariant_ops._matmul_persistent_configs()
        self.assertEqual(
            configs[torch.bfloat16],
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
        )


class TestSegmentedMatmulLaunch(CustomTestCase):
    def test_segment_constants(self):
        """BLOCK_K and the segment count define the reduction order; changing either
        changes the numerics of deterministic mode."""
        self.assertEqual(batch_invariant_ops._SEGMENT_COUNT, 4)
        self.assertEqual(batch_invariant_ops._SEGMENT_BLOCK_K, 64)

    def test_unsplit_launches_are_not_pipelined(self):
        """When one program loops over all segments, software pipelining across the
        loop made the output non-deterministic for some K on MI350X."""
        for N in (32, 512, 1024, 4096, 8192, 8193, 24576, 151936):
            for M in (1, 32, 33, 64, 65, 128, 129, 256, 257, 1024, 2048, 2049, 16384):
                with self.subTest(M=M, N=N):
                    *_, num_stages, split = (
                        batch_invariant_ops._segmented_matmul_launch(M, N)
                    )
                    if not split:
                        self.assertEqual(num_stages, 1)


class TestMatmulPersistentDispatch(CustomTestCase):
    def test_column_slice_takes_one_kernel_for_every_batch_size(self):
        """The DeepGEMM and Triton paths round the bias differently, so the rows of a
        column-slice input must reach the same path whatever the number of rows."""
        K, N = 48, 64
        x = torch.randn(8, K + 32, dtype=torch.bfloat16)[:, :K]
        weight_t = torch.randn(N, K, dtype=torch.bfloat16).t()
        bias = torch.randn(N, dtype=torch.bfloat16)
        paths = []

        def fake_kernel(name):
            def run(a, b, out_dtype, bias=None):
                paths.append(name)
                return torch.empty(a.shape[0], b.shape[1], dtype=out_dtype)

            return run

        with patch.multiple(
            batch_invariant_ops,
            ENABLE_JIT_DEEPGEMM=True,
            _ENABLE_MM_DEEPGEMM=True,
            _is_gfx95=False,
            _matmul_persistent_deepgemm=fake_kernel("deepgemm"),
            _matmul_persistent_triton=fake_kernel("triton"),
        ):
            for batch_size in (1, 2, 8):
                batch_invariant_ops.matmul_persistent(
                    a=x[:batch_size], b=weight_t, bias=bias
                )
        self.assertEqual(paths, ["deepgemm"] * 3)


if __name__ == "__main__":
    unittest.main()
