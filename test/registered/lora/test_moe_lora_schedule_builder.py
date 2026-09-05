"""The packed tile schedule must enumerate exactly the valid tiles.

`build_dual_stage_schedules_masked` is the only producer of what the direct-schedule
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

from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
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

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


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
        return build_dual_stage_schedules_masked(
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

    def test_dual_builder_reuses_graph_outputs(self):
        """CUDA-graph replay needs stable addresses, so the builder must write
        into the caller's buffers instead of allocating new ones.
        """
        rows = [5, 0, 96, 1, 129, 64]
        masked_m = torch.tensor(rows, dtype=torch.int32, device=self.device)
        common = dict(m_max=256, token_width=8, output_width=128)
        buffers = build_dual_stage_schedules_masked(
            masked_m,
            n_gemm1=512,
            n_gemm2=256,
            **common,
        )
        pointers = tuple(tensor.data_ptr() for tensor in buffers)
        masked_m.copy_(
            torch.tensor([0, 8, 17, 64, 3, 1], dtype=torch.int32, device=self.device)
        )
        reused = build_dual_stage_schedules_masked(
            masked_m,
            n_gemm1=512,
            n_gemm2=256,
            schedule1_out=buffers[0],
            tiles1_out=buffers[1],
            schedule2_out=buffers[2],
            tiles2_out=buffers[3],
            **common,
        )
        self.assertEqual(tuple(tensor.data_ptr() for tensor in reused), pointers)
        count = int(reused[1].item())
        expected = self._expected_entries(masked_m.cpu().tolist(), 8, 4)
        decoded = [_decode(int(v)) for v in reused[0][:count].tolist()]
        self.assertEqual(len(decoded), len(set(decoded)))
        self.assertEqual(set(decoded), expected)

    def test_packing_round_trips_at_the_top_of_the_output_field(self):
        """Every representable output-cluster index survives the round trip.

        The output field is the top one, ending one bit below int64's sign bit,
        so saturating it is where a packed word would go negative. One expert
        with one token cluster, so the entry count IS the field width; decoded
        on device because 2^21 entries is slow as a Python list.
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
        values = schedule[:count]
        self.assertGreaterEqual(
            int(values.min().item()), 0, "a packed word went negative"
        )
        zeros = torch.zeros(count, dtype=torch.int64, device=values.device)
        expected_oc = torch.arange(count, dtype=torch.int64, device=values.device)
        for actual, expected in (
            (values & EXPERT_MASK, zeros),
            ((values >> TOKEN_CLUSTER_SHIFT) & TOKEN_CLUSTER_MASK, zeros),
            ((values >> OUTPUT_CLUSTER_SHIFT) & OUTPUT_CLUSTER_MASK, expected_oc),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_geometry_the_packing_cannot_represent(self):
        """The ABI ceilings are enforced, checked through the pure validator.

        Through ``dual_stage_schedule_capacities_masked`` rather than the builder
        because it takes the expert count as an int: materialising a
        2^20-element tensor to prove 2^20 + 1 is refused would be an expensive
        way to test nothing.
        """
        common = dict(token_width=8, n_gemm1=128, n_gemm2=128, output_width=128)
        # m_max at token width 8 needs more token clusters than the field holds.
        with self.assertRaisesRegex(ValueError, "token clusters"):
            dual_stage_schedule_capacities_masked(
                num_experts=4, m_max=(MAX_TOKEN_CLUSTERS + 1) * 8, **common
            )
        # The largest representable expert count, then one past it.
        dual_stage_schedule_capacities_masked(
            num_experts=MAX_EXPERTS, m_max=256, **common
        )
        with self.assertRaisesRegex(ValueError, "expert"):
            dual_stage_schedule_capacities_masked(
                num_experts=MAX_EXPERTS + 1, m_max=256, **common
            )
        # Fields can individually fit while the worst-case capacity overflows
        # the kernel's int32 prefix arithmetic.
        with self.assertRaisesRegex(ValueError, "capacity"):
            dual_stage_schedule_capacities_masked(
                num_experts=1024,
                m_max=1024 * 8,
                token_width=8,
                n_gemm1=MAX_OUTPUT_CLUSTERS * 128,
                n_gemm2=128,
                output_width=128,
            )
        # One output cluster past the usable field width.
        with self.assertRaisesRegex(ValueError, "output clusters"):
            dual_stage_schedule_capacities_masked(
                num_experts=4,
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
            build_dual_stage_schedules_masked(
                masked_m, cluster_shape_mn=(2, 1), **common
            )
        with self.assertRaisesRegex(ValueError, "cluster"):
            build_dual_stage_schedules_masked(masked_m, use_2cta_instrs=True, **common)

    def test_packed_word_is_int64_and_round_trips_past_the_old_int32_fields(self):
        """The widened ABI must survive the trip through the device buffer.

        4096 output clusters against the old int32 layout's cap of 2048, so
        this fails outright on that ABI rather than passing for a wrong reason.
        """
        rows = [3, 0, 7]
        token_width, output_width = 8, 128
        out_clusters2 = 4096
        _schedule1, _tiles1, schedule2, tiles2 = self._build(
            rows,
            m_max=256,
            token_width=token_width,
            n_gemm1=output_width,  # 1 output cluster keeps stage 1 small
            n_gemm2=out_clusters2 * output_width,
            output_width=output_width,
        )
        self.assertEqual(schedule2.dtype, torch.int64)
        # Packed words stay non-negative: the ABI leaves bit 63 clear so the
        # host can read entries back without sign games.
        self.assertGreaterEqual(int(schedule2.min().item()), 0)

        entries = schedule2[: int(tiles2[0].item())].tolist()
        self.assertEqual(
            {_decode(int(word)) for word in entries},
            self._expected_entries(rows, token_width, out_clusters2),
        )
        # Confirm the field that overflows the old layout is really exercised.
        self.assertGreater(max(_decode(int(word))[2] for word in entries), 2047)


if __name__ == "__main__":
    unittest.main()
