"""The packed tile schedule must enumerate exactly the valid tiles.

`build_dual_stage_schedules` is the only producer of what the direct-schedule
CuTeDSL GEMM treats as its work list, and a wrong schedule is SILENT
corruption: tiles pointing at rows the GEMM considers padding, or valid rows
never scheduled, with no device assertion anywhere. Nothing else in the suite
covers it — the provider-level tests run one uniform-occupancy shape on SM100,
so a ragged `masked_m`, a zero-row expert, the ABI ceilings, and the
ordering-heuristic branch were all unguarded (gate-2 review).

Triton only: no SM100 requirement, so this also runs on the H200 pod.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_abi import (
    EXPERT_MASK,
    OUTPUT_CLUSTER_MASK,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_MASK,
    TOKEN_CLUSTER_SHIFT,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
    MAX_OUTPUT_CLUSTERS,
    MAX_TOKEN_CLUSTERS,
    build_dual_stage_schedules,
    build_single_stage_schedule,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _decode(packed: int) -> tuple[int, int, int]:
    """Decode with the ABI constants shared by builder and device scheduler."""
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

    def _build(self, rows, *, m_max, token_width, n_gemm1, n_gemm2, output_width=128):
        masked_m = torch.tensor(rows, dtype=torch.int32, device=self.device)
        return build_dual_stage_schedules(
            masked_m,
            m_max=m_max,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=output_width,
        )

    def _expected_entries(self, rows, token_width, out_clusters):
        """Every (expert, token_cluster, output_cluster) the GEMM must cover."""
        expected = set()
        for expert, valid in enumerate(rows):
            for token_cluster in range((valid + token_width - 1) // token_width):
                for output_cluster in range(out_clusters):
                    expected.add((expert, token_cluster, output_cluster))
        return expected

    def test_ragged_occupancy_enumerates_exactly_the_valid_tiles(self):
        """The set of scheduled tiles equals the set the row counts imply.

        Ragged on purpose, including a zero-row expert (which must contribute
        NO tiles at all — the kernel's contract is that no CTA is launched
        wholly outside an expert's valid rows) and a partial trailing cluster.
        """
        rows = [5, 0, 96, 1, 129, 64]
        token_width, output_width = 8, 128
        n_gemm1, n_gemm2 = 512, 256
        schedule1, tiles1, schedule2, tiles2 = self._build(
            rows,
            m_max=256,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=output_width,
        )

        for schedule, tiles, n_out in (
            (schedule1, tiles1, n_gemm1),
            (schedule2, tiles2, n_gemm2),
        ):
            out_clusters = (n_out + output_width - 1) // output_width
            expected = self._expected_entries(rows, token_width, out_clusters)
            count = int(tiles[0].item())
            self.assertEqual(count, len(expected))
            decoded = [_decode(int(v)) for v in schedule[:count].tolist()]
            # A multiset comparison, so a duplicated tile fails even though the
            # count would still match a set comparison.
            self.assertEqual(len(decoded), len(set(decoded)))
            self.assertEqual(set(decoded), expected)
            self.assertNotIn(1, {entry[0] for entry in decoded})

    def test_ordering_sweeps_the_shorter_axis_fastest(self):
        """Consecutive entries share the longer axis; the shorter one varies.

        This is the L2 rule: the operand re-read across concurrently resident
        clusters must be the one with fewer tiles, so it stays cache-resident.
        Both branches of the heuristic are exercised by choosing geometries
        where token clusters are fewer, then more, than output clusters.
        """
        output_width = 128
        # 2 token clusters vs 4 output clusters -> token axis is shorter.
        _, _, schedule, tiles = self._build(
            [16],
            m_max=256,
            token_width=8,
            n_gemm1=128,
            n_gemm2=512,
            output_width=output_width,
        )
        entries = [_decode(int(v)) for v in schedule[: int(tiles[0].item())].tolist()]
        self.assertEqual(
            [(e[1], e[2]) for e in entries],
            [(tc, oc) for oc in range(4) for tc in range(2)],
        )

        # 8 token clusters vs 1 output cluster -> output axis is shorter.
        _, _, schedule, tiles = self._build(
            [64],
            m_max=256,
            token_width=8,
            n_gemm1=128,
            n_gemm2=128,
            output_width=output_width,
        )
        entries = [_decode(int(v)) for v in schedule[: int(tiles[0].item())].tolist()]
        self.assertEqual(
            [(e[1], e[2]) for e in entries],
            [(tc, oc) for tc in range(8) for oc in range(1)],
        )

    def test_single_stage_matches_dual_and_reuses_graph_outputs(self):
        """The single API preserves packing while omitting the dummy stage."""
        rows = [5, 0, 96, 1, 129, 64]
        masked_m = torch.tensor(rows, dtype=torch.int32, device=self.device)
        common = dict(m_max=256, token_width=8, output_width=128)
        dual1, count1, dual2, count2 = build_dual_stage_schedules(
            masked_m,
            n_gemm1=512,
            n_gemm2=256,
            **common,
        )
        single1, single_count1 = build_single_stage_schedule(
            masked_m,
            n_gemm=512,
            **common,
        )
        single2, single_count2 = build_single_stage_schedule(
            masked_m,
            n_gemm=256,
            **common,
        )
        for dual, dual_count, single, single_count in (
            (dual1, count1, single1, single_count1),
            (dual2, count2, single2, single_count2),
        ):
            count = int(dual_count.item())
            self.assertEqual(int(single_count.item()), count)
            torch.testing.assert_close(single[:count], dual[:count], rtol=0, atol=0)

        schedule_ptr = single1.data_ptr()
        count_ptr = single_count1.data_ptr()
        masked_m.copy_(
            torch.tensor([0, 8, 17, 64, 3, 1], dtype=torch.int32, device=self.device)
        )
        reused, reused_count = build_single_stage_schedule(
            masked_m,
            n_gemm=512,
            schedule_out=single1,
            tiles_out=single_count1,
            **common,
        )
        self.assertEqual(reused.data_ptr(), schedule_ptr)
        self.assertEqual(reused_count.data_ptr(), count_ptr)
        count = int(reused_count.item())
        expected = self._expected_entries(masked_m.cpu().tolist(), 8, 4)
        decoded = [_decode(int(v)) for v in reused[:count].tolist()]
        self.assertEqual(len(decoded), len(set(decoded)))
        self.assertEqual(set(decoded), expected)

    def test_packing_round_trips_at_the_top_of_the_output_field(self):
        """Every representable output-cluster index survives the round trip.

        The output field sits at bit 20, so its 12th bit would be int32's sign
        bit; the constant caps it at 11 usable bits precisely so no packed word
        goes negative. This drives the REAL builder at the top of that range
        (cheap: one expert, one token cluster) and decodes every entry.
        """
        output_width = 128
        out_clusters = MAX_OUTPUT_CLUSTERS  # indices 0 .. MAX-1
        _, _, schedule, tiles = self._build(
            [8],
            m_max=8,
            token_width=8,
            n_gemm1=output_width,
            n_gemm2=out_clusters * output_width,
            output_width=output_width,
        )
        count = int(tiles[0].item())
        self.assertEqual(count, out_clusters)
        values = schedule[:count].tolist()
        self.assertTrue(all(v >= 0 for v in values), "a packed word went negative")
        self.assertEqual(
            [_decode(int(v)) for v in values],
            [(0, 0, oc) for oc in range(out_clusters)],
        )

    def test_rejects_geometry_the_packing_cannot_represent(self):
        masked_m = torch.zeros(4, dtype=torch.int32, device=self.device)
        common = dict(token_width=8, n_gemm1=128, n_gemm2=128, output_width=128)
        # m_max at token width 8 needs more token clusters than the field holds.
        with self.assertRaisesRegex(ValueError, "token clusters"):
            build_dual_stage_schedules(
                masked_m, m_max=(MAX_TOKEN_CLUSTERS + 1) * 8, **common
            )
        # 1024 experts (indices 0..1023) is exactly representable; 1025 is not.
        build_dual_stage_schedules(
            torch.zeros(1024, dtype=torch.int32, device=self.device),
            m_max=256,
            **common,
        )
        with self.assertRaisesRegex(ValueError, "expert"):
            build_dual_stage_schedules(
                torch.zeros(1025, dtype=torch.int32, device=self.device),
                m_max=256,
                **common,
            )
        # Fields can individually fit while the worst-case capacity overflows
        # the kernel's int32 prefix arithmetic.
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_dual_stage_schedules(
                torch.zeros(1024, dtype=torch.int32, device=self.device),
                m_max=1024 * 8,
                token_width=8,
                n_gemm1=MAX_OUTPUT_CLUSTERS * 128,
                n_gemm2=128,
                output_width=128,
            )
        # One output cluster past the usable field width.
        with self.assertRaisesRegex(ValueError, "output clusters"):
            build_dual_stage_schedules(
                masked_m,
                m_max=256,
                token_width=8,
                n_gemm1=128,
                n_gemm2=(MAX_OUTPUT_CLUSTERS + 1) * 128,
                output_width=128,
            )

    def test_rejects_nontrivial_cluster_config(self):
        """Only a 1x1 cluster with 1-CTA MMA is representable.

        The builder emits CTA-tile-granularity indices that the device expands
        by cluster_shape_mn; any other cluster would over-enumerate silently.
        """
        masked_m = torch.zeros(4, dtype=torch.int32, device=self.device)
        common = dict(
            m_max=256, token_width=8, n_gemm1=128, n_gemm2=128, output_width=128
        )
        with self.assertRaisesRegex(ValueError, "cluster"):
            build_dual_stage_schedules(masked_m, cluster_shape_mn=(2, 1), **common)
        with self.assertRaisesRegex(ValueError, "cluster"):
            build_dual_stage_schedules(masked_m, use_2cta_instrs=True, **common)


if __name__ == "__main__":
    unittest.main()
