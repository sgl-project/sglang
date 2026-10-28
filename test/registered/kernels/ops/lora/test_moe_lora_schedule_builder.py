"""Packed schedule coverage, caller-owned output reuse, and geometry admission."""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
    EXPERT_MASK,
    MAX_EXPERTS,
    MAX_OUTPUT_CLUSTERS,
    MAX_TOKEN_CLUSTERS,
    OUTPUT_CLUSTER_MASK,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_MASK,
    TOKEN_CLUSTER_SHIFT,
    build_dual_stage_schedules_masked,
    dual_stage_schedule_capacities_masked,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _decode(packed: int) -> tuple[int, int, int]:
    return (
        packed & EXPERT_MASK,
        (packed >> TOKEN_CLUSTER_SHIFT) & TOKEN_CLUSTER_MASK,
        (packed >> OUTPUT_CLUSTER_SHIFT) & OUTPUT_CLUSTER_MASK,
    )


class TestMoeLoraScheduleBuilder(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda")

    def _expected_entries(self, rows, token_width, out_clusters):
        expected = set()
        for expert, valid in enumerate(rows):
            for token_cluster in range((valid + token_width - 1) // token_width):
                for output_cluster in range(out_clusters):
                    expected.add((expert, token_cluster, output_cluster))
        return expected

    def test_ragged_tiles_and_output_reuse(self):
        rows = [5, 0, 96, 1, 129, 64]
        masked_m = torch.tensor(rows, dtype=torch.int32, device=self.device)
        token_width, output_width = 8, 128
        n_gemm1, n_gemm2 = 512, 256
        common = dict(
            m_max=256,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=output_width,
        )
        buffers = build_dual_stage_schedules_masked(masked_m, **common)
        pointers = tuple(tensor.data_ptr() for tensor in buffers)

        for current_rows in (rows, [0, 8, 17, 64, 3, 1]):
            if current_rows is not rows:
                masked_m.copy_(
                    torch.tensor(current_rows, dtype=torch.int32, device=self.device)
                )
                reused = build_dual_stage_schedules_masked(
                    masked_m,
                    schedule1_out=buffers[0],
                    tiles1_out=buffers[1],
                    schedule2_out=buffers[2],
                    tiles2_out=buffers[3],
                    **common,
                )
                for actual, original in zip(reused, buffers):
                    self.assertIs(actual, original)
                self.assertEqual(
                    tuple(tensor.data_ptr() for tensor in reused), pointers
                )

            for schedule, tiles, n_out in (
                (buffers[0], buffers[1], n_gemm1),
                (buffers[2], buffers[3], n_gemm2),
            ):
                out_clusters = (n_out + output_width - 1) // output_width
                expected = self._expected_entries(
                    current_rows, token_width, out_clusters
                )
                count = int(tiles[0].item())
                self.assertEqual(count, len(expected))
                decoded = [_decode(int(v)) for v in schedule[:count].tolist()]
                self.assertEqual(len(decoded), len(set(decoded)))
                self.assertEqual(set(decoded), expected)

    def test_rejects_geometry_the_packing_cannot_represent(self):
        common = dict(
            num_experts=4,
            m_max=256,
            token_width=8,
            n_gemm1=128,
            n_gemm2=128,
            output_width=128,
        )
        dual_stage_schedule_capacities_masked(**{**common, "num_experts": MAX_EXPERTS})
        for overrides, error in (
            ({"num_experts": MAX_EXPERTS + 1}, "expert"),
            ({"m_max": (MAX_TOKEN_CLUSTERS + 1) * 8}, "token clusters"),
            ({"n_gemm2": (MAX_OUTPUT_CLUSTERS + 1) * 128}, "output clusters"),
            (
                {
                    "num_experts": 1024,
                    "m_max": 1024 * 8,
                    "n_gemm1": MAX_OUTPUT_CLUSTERS * 128,
                },
                "capacity",
            ),
            ({"cluster_shape_mn": (2, 1)}, "cluster"),
            ({"use_2cta_instrs": True}, "cluster"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, error),
            ):
                dual_stage_schedule_capacities_masked(**{**common, **overrides})


if __name__ == "__main__":
    unittest.main()
