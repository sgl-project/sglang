"""CPU tests for the launch grid of the fused MoE finalize + shared-expert gate."""

import importlib
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.ops.elementwise import elementwise
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class _Launches:
    def __init__(self):
        self.grids = []

    def kernel(self, name):
        launches = self

        class _Kernel:
            def __getitem__(self, grid):
                def launch(*args, **meta):
                    launches.grids.append((name, grid, meta))

                return launch

        return _Kernel()


def _call(tokens, hidden=2560, top_k=10):
    elementwise.fused_moe_finalize_gate_sigmoid_mul_add(
        torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        torch.zeros(hidden, dtype=torch.bfloat16),
        torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        torch.zeros(tokens * top_k, hidden, dtype=torch.bfloat16),
        torch.zeros(tokens * top_k, dtype=torch.int32),
        torch.zeros(tokens, top_k, dtype=torch.float32),
    )


class TestFusedFinalizeGateColumnChunk(CustomTestCase):
    def tearDown(self):
        importlib.reload(elementwise)

    def test_chunk_must_be_off_or_a_supported_width(self):
        field = envs.SGLANG_MOE_FINALIZE_GATE_COLUMN_CHUNK
        for value, ok in ((0, True), (1024, True), (128, True), (64, False), (300, False)):
            with self.subTest(value=value), field.override(value):
                if ok:
                    importlib.reload(elementwise)
                    self.assertEqual(elementwise._FINALIZE_GATE_COLUMN_CHUNK, value)
                else:
                    with self.assertRaises(ValueError):
                        importlib.reload(elementwise)

    def test_chunked_grid_only_up_to_64_tokens_with_the_row_kernel_config(self):
        launches = _Launches()
        fake_triton = SimpleNamespace(
            cdiv=lambda a, b: -(-a // b),
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        )
        with mock.patch.multiple(
            elementwise,
            triton=fake_triton,
            is_arch_support_pdl=lambda: True,
            _FINALIZE_GATE_COLUMN_CHUNK=1024,
            _fused_finalize_gate_sigmoid_mul_add_kernel=launches.kernel("row"),
            _fused_finalize_gate_sigmoid_mul_add_chunked_kernel=launches.kernel("chunked"),
        ):
            for tokens in (1, 64, 65, 576):
                _call(tokens)
            with mock.patch.object(elementwise, "_FINALIZE_GATE_COLUMN_CHUNK", 0):
                _call(8)
        names = [(name, grid) for name, grid, _ in launches.grids]
        self.assertEqual(
            names,
            [
                ("chunked", (1, math.ceil(2560 / 1024))),
                ("chunked", (64, math.ceil(2560 / 1024))),
                ("row", (65,)),
                ("row", (576,)),
                ("row", (8,)),
            ],
        )
        # The gate reduction is one tree only if the chunked programs keep the
        # row kernel's BLOCK_SIZE and num_warps.
        chunked, row = launches.grids[0][2], launches.grids[4][2]
        for key in ("BLOCK_SIZE", "num_warps", "USE_PDL", "launch_pdl"):
            self.assertEqual(chunked[key], row[key], key)
        self.assertEqual(chunked["CHUNK"], 1024)


if __name__ == "__main__":
    unittest.main()
