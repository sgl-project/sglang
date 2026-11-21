"""What ``w8a8_block_fp8_matmul_triton`` launches, on gfx115x (RDNA3.5) and elsewhere.

A wrong kernel or tile computes the same product, only slower, so no numerical
check sees it. Here recorders stand in for the Triton kernels and the public
function runs on CPU tensors; the assertions are on the launch it makes.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm import fp8_kernel
from sglang.srt.utils import use_gfx115_kernels
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BLOCK = 128
N, K = 336, 384


class _RecordingKernel:
    """Stands in for a Triton kernel: ``kernel[grid](*args, **meta)``."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **meta):
            # Positions 5 to 9 are M, N, K and the scale block.
            shown = grid(meta) if callable(grid) else grid
            self.launches.append((shown, tuple(args[5:10]), meta))

        return launch


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _meta(*, tile, block_k=BLOCK, group=1, warps=4, stages=1, masking=False):
    return {
        "BLOCK_SIZE_M": tile[0],
        "BLOCK_SIZE_N": tile[1],
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group,
        "num_warps": warps,
        "num_stages": stages,
        "needs_masking": masking,
    }


class TestBlockFp8MatmulDispatch(CustomTestCase):
    def _launch(
        self,
        *,
        rows,
        gfx115,
        gfx1250=False,
        sm90=False,
        tuned=None,
        k=K,
        block=(BLOCK, BLOCK),
    ):
        """Launches by kernel, and the output, for one call with ``rows`` leading dimensions of A."""
        rows = (rows,) if isinstance(rows, int) else rows
        kernels = {
            name: _RecordingKernel()
            for name in ("upcast", "shared", "hopper", "split_k_reduce")
        }
        lookups = []

        def tuned_configs(*args):
            lookups.append(args)
            return tuned

        patches = {
            "_is_gfx115": gfx115,
            "_is_gfx1250": gfx1250,
            "_w8a8_block_fp8_matmul_gfx1250": kernels["upcast"],
            "_w8a8_block_fp8_matmul": kernels["shared"],
            "_w8a8_block_fp8_matmul_hopper": kernels["hopper"],
            "_reduce_block_fp8_split_k": kernels["split_k_reduce"],
            "get_w8a8_block_fp8_configs": tuned_configs,
            "get_platform": lambda: SimpleNamespace(is_sm90=sm90),
        }
        with ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(fp8_kernel, name, value))
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                torch.zeros(*rows, k, dtype=torch.float8_e4m3fn),
                torch.zeros(N, k, dtype=torch.float8_e4m3fn),
                torch.ones(*rows, _cdiv(k, block[1])),
                torch.ones(_cdiv(N, block[0]), _cdiv(k, block[1])),
                list(block),
                output_dtype=torch.bfloat16,
            )
        self.assertEqual(lookups, [(N, k, block[0], block[1])])
        self.assertEqual((out.shape, out.dtype), ((*rows, N), torch.bfloat16))
        return {name: kernel.launches for name, kernel in kernels.items()}

    def _only(self, launches, name):
        for other, made in launches.items():
            if other != name:
                self.assertEqual(made, [], other)
        self.assertEqual(len(launches[name]), 1)
        return launches[name][0]

    def test_the_module_gate_is_the_gfx115_predicate(self):
        self.assertIs(fp8_kernel._is_gfx115, use_gfx115_kernels())

    def test_default_launch_follows_the_row_count(self):
        # Both sides of each of the two switches, a count above one row tile, and
        # a full prefill chunk.
        tiles_by_rows = {
            1: (16, 64),
            16: (16, 64),
            17: (32, 64),
            32: (32, 64),
            33: (64, 32),
            70: (64, 32),
            4096: (64, 32),
        }
        for rows, tile in tiles_by_rows.items():
            with self.subTest(rows=rows):
                launch = self._only(self._launch(rows=rows, gfx115=True), "upcast")
                grid = (_cdiv(rows, tile[0]) * _cdiv(N, tile[1]),)
                self.assertEqual(
                    launch, (grid, (rows, N, K, BLOCK, BLOCK), _meta(tile=tile))
                )

    def test_default_launch_for_other_inputs(self):
        # Leading dimensions of A flatten into the row count: 2 x 9 = 18 rows.
        launch = self._only(self._launch(rows=(2, 9), gfx115=True), "upcast")
        self.assertEqual(launch, ((6,), (18, N, K, BLOCK, BLOCK), _meta(tile=(32, 64))))
        # K ends in a partial scale block.
        launch = self._only(self._launch(rows=1, gfx115=True, k=400), "upcast")
        self.assertEqual(
            launch,
            ((6,), (1, N, 400, BLOCK, BLOCK), _meta(tile=(16, 64), masking=True)),
        )
        # A scale block that is not square: the K tile follows its second size,
        # and the N tile does not follow its first.
        for rows, tile in ((1, (16, 64)), (17, (32, 64)), (33, (64, 32))):
            launch = self._only(
                self._launch(rows=rows, gfx115=True, block=(64, 32)), "upcast"
            )
            grid = (_cdiv(rows, tile[0]) * _cdiv(N, tile[1]),)
            self.assertEqual(
                launch, (grid, (rows, N, K, 64, 32), _meta(tile=tile, block_k=32))
            )

    def test_tuned_config_keeps_its_tile_without_pipelining(self):
        tuned = {
            1: _meta(tile=(64, 128), group=8, warps=8, stages=4),
            64: _meta(tile=(32, 32), group=2, warps=4, stages=2),
        }
        for config in tuned.values():
            del config["needs_masking"]
        # The entry with the nearest row count: 17 is nearer to 1, 50 to 64.
        for rows, key, grid in ((17, 1, (3,)), (50, 64, (22,))):
            with self.subTest(rows=rows):
                launch = self._only(
                    self._launch(rows=rows, gfx115=True, tuned=tuned), "upcast"
                )
                meta = {**tuned[key], "num_stages": 1, "needs_masking": False}
                self.assertEqual(launch, (grid, (rows, N, K, BLOCK, BLOCK), meta))
        self.assertEqual((tuned[1]["num_stages"], tuned[64]["num_stages"]), (4, 2))

    def test_other_platforms_launch_what_they_did_before(self):
        default = _meta(tile=(64, BLOCK), group=32, stages=3)
        for rows in (1, 70):
            with self.subTest(platform="gate off", rows=rows):
                launch = self._only(self._launch(rows=rows, gfx115=False), "shared")
                self.assertEqual(launch[0], (_cdiv(rows, 64) * 3,))
                self.assertEqual(launch[2], default)
        with self.subTest(platform="gate off, 64 x 32 scale block"):
            launch = self._only(
                self._launch(rows=1, gfx115=False, block=(64, 32)), "shared"
            )
            self.assertEqual(
                launch,
                (
                    (6,),
                    (1, N, K, 64, 32),
                    _meta(tile=(64, 64), block_k=32, group=32, stages=3),
                ),
            )
        with self.subTest(platform="gfx1250"):
            launch = self._only(
                self._launch(rows=1, gfx115=False, gfx1250=True), "upcast"
            )
            self.assertEqual(launch[2], {**default, "num_stages": 1})
        with self.subTest(platform="gfx1250, tuned"):
            tuned = _meta(tile=(32, 64), group=4, warps=8, stages=3)
            del tuned["needs_masking"]
            launch = self._only(
                self._launch(rows=1, gfx115=False, gfx1250=True, tuned={1: tuned}),
                "upcast",
            )
            self.assertEqual(
                launch[2], {**tuned, "num_stages": 1, "needs_masking": False}
            )
            self.assertEqual(tuned["num_stages"], 3)
        with self.subTest(platform="sm90, tuned with SWAP_AB"):
            tuned = {**default, "SWAP_AB": True}
            del tuned["needs_masking"]
            launch = self._only(
                self._launch(rows=1, gfx115=False, sm90=True, tuned={1: tuned}),
                "hopper",
            )
            self.assertEqual(launch[0], (3, 1))
            self.assertEqual(launch[2], {**tuned, "needs_masking": False})
        with self.subTest(platform="sm90, tuned with SPLIT_K"):
            tuned = {**default, "SPLIT_K": 2}
            del tuned["needs_masking"]
            launches = self._launch(rows=1, gfx115=False, sm90=True, tuned={1: tuned})
            self.assertEqual((launches["upcast"], launches["shared"]), ([], []))
            self.assertEqual(len(launches["hopper"]), 1)
            self.assertEqual(launches["hopper"][0][0], (3, 2))
            self.assertEqual(len(launches["split_k_reduce"]), 1)
        # The Hopper kernel is for SM90 only, whatever the tuned config asks for.
        for key, value in (("SWAP_AB", True), ("SPLIT_K", 2)):
            with self.subTest(platform="not sm90", tuned_key=key):
                tuned = {**default, key: value}
                del tuned["needs_masking"]
                launches = self._launch(rows=1, gfx115=False, tuned={1: tuned})
                self.assertEqual((launches["upcast"], launches["hopper"]), ([], []))
                self.assertEqual(len(launches["shared"]), 1)
                self.assertEqual(launches["split_k_reduce"], [])


if __name__ == "__main__":
    unittest.main()
