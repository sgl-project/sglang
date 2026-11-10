"""CPU tests for the DSV4.1-on-V100 dry-run allocator and expert-spill LRU."""

from __future__ import annotations

import ctypes
import os
import unittest
from io import StringIO
from unittest.mock import patch

from sglang.srt.layers.moe.dsv41_expert_spill import (
    RoutedExpertLru,
    cpu_offload_gb_tax,
    plan_routed_expert_spill,
    spill_host_is_marlin_packed,
    v1_spill_plan,
)
from sglang.srt.mem_cache.dsv41_host_placement import (
    GIB,
    PRE_H1_NODE_TOTAL_GIB,
    EngramNumaError,
    NumaNodeMem,
    mmap_hugetlb,
    mmap_numa_thp,
    plan_engram_host_tables,
    read_node_hugepages,
    round_to_huge_pages,
    smaps_huge_kb,
)
from sglang.srt.mem_cache.dsv41_v100_budget import (
    HBM_FAIL_GIB,
    LANDING_SLOTS_V1,
    TARGET_SLACK_GIB,
    allocate,
    engram_table_bytes,
    landing_slots_for_dspark,
    main,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _nodes(totals=None):
    totals = totals or PRE_H1_NODE_TOTAL_GIB
    return {
        n: NumaNodeMem(n, int(round(g * GIB)), int(round(g * GIB)))
        for n, g in totals.items()
    }


class TestDsv41V100Budget(CustomTestCase):
    def test_v1_host_engram_and_spill_fits_hbm(self):
        result = allocate(allow_numa_split=True)
        self.assertTrue(result.hbm_ok)
        self.assertTrue(result.slack_ok)
        for row in result.ranks:
            self.assertLessEqual(row.hbm_gib, HBM_FAIL_GIB)
            self.assertGreaterEqual(row.slack_gib, TARGET_SLACK_GIB)
            self.assertEqual(row.engram_hbm_gib, 0.0)
            self.assertAlmostEqual(row.spill_gib, 10.0, places=2)

    def test_chunk_4k_still_has_slack(self):
        result = allocate(chunk=4096, allow_numa_split=True)
        self.assertTrue(result.hbm_ok)
        self.assertTrue(result.slack_ok)

    def test_no_spill_exceeds_fail_line(self):
        result = allocate(spill_gib=0.0, allow_numa_split=True)
        self.assertFalse(result.hbm_ok)
        self.assertGreater(max(r.hbm_gib for r in result.ranks), HBM_FAIL_GIB)

    def test_engram_on_hbm_exceeds_fail_line(self):
        result = allocate(host_engram=False, spill_gib=10.0)
        self.assertFalse(result.hbm_ok)

    def test_lie_experts_kept_fails_script(self):
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            rc = main(["--allow-numa-split", "--lie-experts-kept-gib", "40"])
        self.assertEqual(rc, 1)

    def test_honest_script_with_split_exits_zero(self):
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            rc = main(["--allow-numa-split"])
        self.assertEqual(rc, 0)

    def test_node_local_engram_fails_loud_on_pre_h1_nodes(self):
        with self.assertRaises(EngramNumaError):
            allocate(
                allow_numa_split=False, node_totals_gib=PRE_H1_NODE_TOTAL_GIB
            )

    def test_post_h1_node_local_engram_fits(self):
        result = allocate(allow_numa_split=False, engram_layout="shared", spill_gib=10.0)
        self.assertFalse(result.host.crosses_upi)
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            self.assertEqual(main([]), 0)

    def test_split_places_overflow_off_gpu_node(self):
        result = allocate(
            allow_numa_split=True,
            engram_layout="private",
            node_totals_gib=PRE_H1_NODE_TOTAL_GIB,
        )
        self.assertTrue(result.host.crosses_upi)
        self.assertTrue(any("UPI" in w or "cross" in w.lower() for w in result.host.warnings))

    def test_v1_counts_landing_six_and_zero_dspark_rows(self):
        result = allocate(allow_numa_split=True)
        row = result.ranks[0]
        self.assertEqual(result.landing_slots, LANDING_SLOTS_V1)
        self.assertAlmostEqual(row.landing_gib, 107.6 / 1024.0, places=3)
        self.assertEqual(row.dspark_experts_gib, 0.0)
        self.assertEqual(row.draft_graphs_gib, 0.0)
        # Pre-landing table was 28.47; 6-slot pool is the measured 107.6 MiB.
        self.assertAlmostEqual(row.hbm_gib, 28.47 + 107.6 / 1024.0, places=2)

    def test_with_dspark_does_not_fold_mtp_into_spilled_routed(self):
        v1 = allocate(allow_numa_split=True)
        ds = allocate(allow_numa_split=True, skip_dspark=False)
        self.assertAlmostEqual(
            v1.ranks[0].experts_kept_gib, ds.ranks[0].experts_kept_gib, places=3
        )
        self.assertEqual(ds.landing_slots, landing_slots_for_dspark(5))
        self.assertEqual(ds.landing_slots, 36)
        self.assertGreater(ds.ranks[0].dspark_experts_gib, 0.7)
        self.assertLess(ds.ranks[0].dspark_experts_gib, 1.2)
        self.assertGreater(ds.ranks[0].graphs_gib, v1.ranks[0].graphs_gib)
        self.assertGreater(ds.ranks[0].draft_graphs_gib, 0.0)
        self.assertAlmostEqual(ds.ranks[0].landing_gib, 6.0 * v1.ranks[0].landing_gib, places=3)

    def test_with_dspark_spill10_misses_slack_spill12_meets(self):
        tight = allocate(allow_numa_split=True, skip_dspark=False, spill_gib=10.0)
        wide = allocate(allow_numa_split=True, skip_dspark=False, spill_gib=12.0)
        self.assertTrue(tight.hbm_ok)
        self.assertFalse(tight.slack_ok)
        self.assertTrue(wide.hbm_ok)
        self.assertTrue(wide.slack_ok)
        self.assertGreaterEqual(wide.ranks[0].slack_gib, TARGET_SLACK_GIB)

    def test_with_dspark_script_requires_slack(self):
        with patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()):
            rc10 = main(["--allow-numa-split", "--with-dspark", "--spill-gb", "10"])
            rc12 = main(["--allow-numa-split", "--with-dspark", "--spill-gb", "12"])
            rc10_ok = main(
                [
                    "--allow-numa-split",
                    "--with-dspark",
                    "--spill-gb",
                    "10",
                    "--allow-miss-slack",
                ]
            )
        self.assertEqual(rc10, 1)
        self.assertEqual(rc12, 0)
        self.assertEqual(rc10_ok, 0)


class TestEngramHostPlacement(CustomTestCase):
    def test_tiny_table_fits_node_local(self):
        plan = plan_engram_host_tables(
            table_nbytes=((1, 1 * GIB),),
            layout="shared",
            tp_size=8,
            allow_split=False,
            nodes=_nodes(),
            huge_pages_total=0,
        )
        self.assertEqual(plan.mappings[0].node, 1)
        self.assertFalse(plan.crosses_upi)
        self.assertEqual(plan.huge_pages_total, 0)

    def test_189_gib_does_not_fit_node1(self):
        tables = ((1, int(94.5 * GIB)), (14, int(94.5 * GIB)))
        with self.assertRaises(EngramNumaError):
            plan_engram_host_tables(
                table_nbytes=tables,
                layout="shared",
                tp_size=8,
                allow_split=False,
                nodes=_nodes(),
                huge_pages_total=0,
            )

    def test_split_puts_second_layer_on_node0(self):
        tables = ((1, int(94.5 * GIB)), (14, int(94.5 * GIB)))
        plan = plan_engram_host_tables(
            table_nbytes=tables,
            layout="shared",
            tp_size=8,
            allow_split=True,
            extra_per_rank_bytes=0,
            nodes=_nodes(),
            huge_pages_total=0,
        )
        nodes = {m.layer_id: m.node for m in plan.mappings}
        self.assertEqual(nodes[1], 1)
        self.assertEqual(nodes[14], 0)
        self.assertTrue(plan.crosses_upi)

    def test_engram_table_bytes_match_fit_note(self):
        total = engram_table_bytes(384_006_168) + engram_table_bytes(384_016_682)
        self.assertAlmostEqual(total / GIB, 188.8, places=1)


class TestRoutedExpertLru(CustomTestCase):
    def test_plan_spills_only_routed_rows(self):
        plan = plan_routed_expert_spill(
            spill_gib=10.0,
            local_routed=48,
            bytes_per_expert=18 * 1024**2,
            n_shared=1,
            n_layers=40,
        )
        self.assertGreater(plan.n_spilled, 0)
        self.assertLess(plan.n_spilled, 48)
        self.assertEqual(plan.n_kept_routed + plan.n_spilled, 48)
        self.assertEqual(plan.n_shared, 1)

    def test_v1_plan_is_about_ten_gib(self):
        plan = v1_spill_plan(10.0)
        self.assertGreater(plan.spill_gib, 8.0)

    def test_dspark_draft_moe_does_not_inherit_target_spill(self):
        """12 GiB spill is for 384-expert target, not 128-expert draft."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.dsv41_expert_spill import plan_gpu_expert_slots

        with envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.override(True), envs.SGLANG_DSV41_EXPERT_SPILL_GB.override(
            12.0
        ):
            gpu_n, plan = plan_gpu_expert_slots(
                num_local_experts=16,
                n_shared=0,
                hidden_size=5120,
                intermediate_size_per_partition=2560,
            )
        self.assertEqual(gpu_n, 16)
        self.assertIsNone(plan)

        with envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.override(True), envs.SGLANG_DSV41_EXPERT_SPILL_GB.override(
            12.0
        ):
            gpu_n, plan = plan_gpu_expert_slots(
                num_local_experts=49,
                n_shared=1,
                hidden_size=5120,
                intermediate_size_per_partition=2560,
            )
        self.assertIsNotNone(plan)
        self.assertGreater(gpu_n, 1)
        self.assertLess(plan.n_spilled, 48)
        self.assertLessEqual(plan.spill_gib, 12.0)

    def test_lru_shrink_and_ensure(self):
        import torch

        weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(weight.clone(), n_shared=1, n_spilled=3)
        self.assertEqual(lru.n_kept_routed, 4)
        shrunk = lru.apply_shrink()
        self.assertEqual(shrunk.shape[0], 5)  # 4 kept routed + 1 shared
        self.assertLess(lru.gpu_bytes, weight.numel() * 4)
        lru.ensure([6])  # spilled routed id
        rows = lru.gather_rows([6, 7])
        self.assertEqual(tuple(rows.shape), (2, 4))
        # shared expert (id 7) is the last logical row
        self.assertTrue(torch.equal(rows[1], weight[7]))

    def test_lru_siblings_and_map_ids_stay_aligned(self):
        import torch

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        w2 = (torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2) + 100)
        lru = RoutedExpertLru(
            w13.clone(), n_shared=1, n_spilled=3, siblings=[w2.clone()]
        )
        lru.apply_shrink()
        self.assertEqual(lru.shrunk_tensors()[0].shape[0], 5)
        self.assertEqual(lru.shrunk_tensors()[1].shape[0], 5)
        lru.ensure([6])
        ids = torch.tensor([[6, 7, -1]], dtype=torch.int32)
        mapped = lru.map_ids(ids)
        self.assertEqual(int(mapped[0, 2]), -1)
        slots = lru.physical_ids([6, 7])
        self.assertEqual(int(mapped[0, 0]), slots[0])
        self.assertEqual(int(mapped[0, 1]), slots[1])
        # w2 row for logical 6 matches the original after ensure
        phys = slots[0]
        self.assertTrue(torch.equal(lru.shrunk_tensors()[1][phys], w2[6]))
        again = lru.map_ids(ids)
        self.assertTrue(torch.equal(mapped, again))
        lru.ensure([5])
        mapped5 = lru.map_ids(torch.tensor([[5, 6]], dtype=torch.int32))
        self.assertEqual(int(mapped5[0, 0]), lru.physical_ids([5])[0])
        self.assertEqual(int(mapped5[0, 1]), lru.physical_ids([6])[0])

    def test_cold_set_placement_gathers_original_rows(self):
        """A non-tail cold set; every id must still gather its own row."""
        import random

        import torch

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        w2 = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2) + 100
        cold = [1, 4, 5]  # routed ids 0..6, shared 7
        lru = RoutedExpertLru(
            w13.clone(), n_shared=1, n_spilled=3, siblings=[w2.clone()], cold_ids=cold
        )
        self.assertEqual(lru._kept_ids, [0, 2, 3, 6])
        lru.apply_shrink()
        g13, g2 = lru.shrunk_tensors()
        self.assertEqual(g13.shape[0], 5)
        # kept rows sit at their kept index; shared after them
        for slot, e in enumerate([0, 2, 3, 6]):
            self.assertTrue(torch.equal(g13[slot], w13[e]))
        self.assertTrue(torch.equal(g13[4], w13[7]))
        rng = random.Random(0)
        for _ in range(60):
            ids = rng.sample(range(7), 3) + [7]
            rows = lru.gather_rows(ids)
            for r, e in zip(rows, ids):
                self.assertTrue(torch.equal(r, w13[e]), (ids, e))
            slots = lru.physical_ids(ids)
            for s, e in zip(slots, ids):
                self.assertTrue(torch.equal(g2[s], w2[e]), (ids, e))

    def test_cold_first_victim_keeps_hot_set_resident(self):
        import torch

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3, cold_ids=[4, 5, 6])
        lru.apply_shrink()
        lru.ensure([4])  # first cold miss evicts the oldest kept expert (0)
        self.assertNotIn(0, lru._id_to_slot)
        lru.ensure([1, 2, 3])  # touch the remaining kept ones
        lru.ensure([5])  # 4 is resident and cold-homed -> it is the victim, not a kept one
        self.assertNotIn(4, lru._id_to_slot)
        for e in (1, 2, 3, 5):
            self.assertIn(e, lru._id_to_slot)
        self.assertEqual(lru.n_swaps, 2)
        # rows are still correct after the churn
        for e in range(7):
            self.assertTrue(torch.equal(lru.gather_rows([e])[0], w13[e]))

    def test_ensure_unique_exceeds_slots_thrashes_not_raises(self):
        """Prefill unique(batch) > n_kept_routed must not refuse."""
        import torch

        # 3 routed + 1 shared; 2 GPU routed slots, 1 spilled host row.
        w13 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=1)
        self.assertEqual(lru.n_kept_routed, 2)
        lru.apply_shrink()
        lru.ensure([0, 1, 2])
        self.assertIn(2, lru._id_to_slot)
        self.assertTrue(torch.equal(lru.gather_rows([2])[0], w13[2]))
        self.assertGreater(lru.n_thrash, 0)
        # Rows survive the churn; batch exactness is the multi-pass test below.
        for e in (0, 1, 2):
            self.assertTrue(torch.equal(lru.gather_rows([e])[0], w13[e]))

    def test_prefill_over_slots_runs_every_expert_choice_once(self):
        """A prefill batch needing more experts than GPU slots stays exact.

        One ensure over the whole batch evicted earlier ids of the same batch
        and their tokens mapped to -1, silently dropping those experts.
        """
        from typing import NamedTuple

        import torch

        from sglang.srt.layers.moe.dsv41_expert_spill import (
            run_moe_with_expert_spill,
        )

        class _TopK(NamedTuple):
            topk_weights: torch.Tensor
            topk_ids: torch.Tensor

        class _Dispatch(NamedTuple):
            topk_output: _TopK

        class _Combine(NamedTuple):
            hidden_states: torch.Tensor

        # 6 routed + 1 shared (id 6); 3 spilled leaves 3 GPU routed slots.
        rows = torch.tensor([1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]).reshape(7, 1)
        lru = RoutedExpertLru(rows.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        moe = type("Moe", (), {})()
        moe._dsv41_expert_lru = lru

        def apply(layer, dispatch_output):
            ids = dispatch_output.topk_output.topk_ids
            weights = dispatch_output.topk_output.topk_weights
            gpu = layer._dsv41_expert_lru._gpu[:, 0]
            picked = torch.where(ids >= 0, gpu[ids.clamp(min=0).long()], 0.0)
            return _Combine(hidden_states=(picked * weights).sum(dim=1))

        ids = torch.tensor([[0, 3, 6], [1, 4, 6], [2, 5, 6]] * 3, dtype=torch.int32)
        weights = torch.full(ids.shape, 1.0)
        out = run_moe_with_expert_spill(
            moe, _Dispatch(_TopK(weights, ids.clone())), apply
        )
        expected = rows[ids.long(), 0].sum(dim=1)
        self.assertTrue(torch.equal(out.hidden_states, expected))

    def test_prefill_landing_passes_run_every_choice_once(self):
        """Prefill through the landing pool stays exact and leaves GPU residency
        alone: kept ids run in one pass only, and each landing pass maps only
        its own host rows, including when an unused host row splits the runs."""
        from typing import NamedTuple

        import torch

        from sglang.srt.layers.moe.dsv41_expert_spill import (
            SpillLandingPool,
            run_moe_with_expert_spill,
        )

        class _TopK(NamedTuple):
            topk_weights: torch.Tensor
            topk_ids: torch.Tensor

        class _Dispatch(NamedTuple):
            topk_output: _TopK

        class _Combine(NamedTuple):
            hidden_states: torch.Tensor

        rows = torch.tensor([1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]).reshape(7, 1)
        lru = RoutedExpertLru(rows.clone(), n_shared=1, n_spilled=4)
        lru.apply_shrink()
        pool = SpillLandingPool(
            n_landing=2,
            tensors={"w13_weight": torch.zeros(2, 1)},
            attrs=["w13_weight"],
            dst_ptrs=torch.zeros(1),
            row_bytes=torch.zeros(1),
            slot_host_row=torch.zeros(2),
            land_ids=None,
            quant_info=None,
        )
        moe = type("Moe", (), {})()
        moe._dsv41_expert_lru = lru
        moe._dsv41_landing_pool = pool
        moe._dsv41_spill_host = {"w13_weight": lru._host}

        def apply(layer, dispatch_output):
            ids = dispatch_output.topk_output.topk_ids
            weights = dispatch_output.topk_output.topk_weights
            gpu = layer._dsv41_expert_lru._gpu[:, 0]
            picked = torch.where(ids >= 0, gpu[ids.clamp(min=0).long()], 0.0)
            land = layer._dsv41_land_ids
            landed = pool.tensors["w13_weight"][:, 0]
            picked = picked + torch.where(
                land >= 0, landed[land.clamp(min=0).long()], 0.0
            )
            return _Combine(hidden_states=(picked * weights).sum(dim=1))

        # Host rows 0, 2, 3 are used (row 1 is not); two landing slots -> two passes.
        ids = torch.tensor([[0, 2, 6], [1, 4, 6], [5, 1, 6]] * 3, dtype=torch.int32)
        weights = torch.full(ids.shape, 1.0)
        out = run_moe_with_expert_spill(
            moe, _Dispatch(_TopK(weights, ids.clone())), apply
        )
        expected = rows[ids.long(), 0].sum(dim=1)
        self.assertTrue(torch.equal(out.hidden_states, expected))
        self.assertEqual(lru.n_swaps, 0)

    def test_ensure_prefers_non_needed_occupant_before_thrash(self):
        import torch

        w13 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=1)
        lru.apply_shrink()
        # 0,1 start on GPU; 2 is spilled. Loading 0+2 must evict 1, not 0.
        lru.ensure([0, 2])
        self.assertIn(0, lru._id_to_slot)
        self.assertIn(2, lru._id_to_slot)
        self.assertNotIn(1, lru._id_to_slot)
        self.assertEqual(lru.n_thrash, 0)

    def test_cold_set_rejects_bad_ids(self):
        import torch

        w13 = torch.zeros(8, 4)
        with self.assertRaises(ValueError):
            RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3, cold_ids=[1, 1, 2])
        with self.assertRaises(ValueError):
            RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3, cold_ids=[1, 2, 7])

    def test_spill_placement_from_table_and_tail_fallback(self):
        import tempfile

        import torch
        from torch import nn

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.dsv41_expert_spill import (
            RoutedExpertSpillPlan,
            remap_shared_expert_gpu_index,
            spill_placement,
        )

        plan = RoutedExpertSpillPlan(
            local_routed=6, n_shared=1, n_kept_routed=4, n_spilled=2,
            bytes_per_expert=1, spill_bytes=2, kept_bytes=4,
        )
        # coldest-first order per (layer, rank); layer 0 rank 1 -> cold {5, 0}
        table = torch.zeros(2, 2, 6, dtype=torch.int64)
        table[0, 1] = torch.tensor([5, 0, 3, 1, 2, 4])
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"cold_ids": table}, f.name)
            with envs.SGLANG_DSV41_EXPERT_SPILL_COLD_SET.override(f.name):
                moe = nn.Module()
                moe._dsv41_expert_spill_plan = plan
                moe._num_local_routed = 6
                moe.layer_id = 0
                moe.moe_ep_rank = 1
                kept, cold = spill_placement(moe)
                self.assertEqual(cold, [0, 5])
                self.assertEqual(kept, [1, 2, 3, 4])
                self.assertEqual(remap_shared_expert_gpu_index(moe, 3), 2)
                self.assertEqual(remap_shared_expert_gpu_index(moe, 6), 4)  # shared
                with self.assertRaises(KeyError):
                    remap_shared_expert_gpu_index(moe, 5)
                # layer outside the table -> tail
                moe2 = nn.Module()
                moe2._dsv41_expert_spill_plan = plan
                moe2._num_local_routed = 6
                moe2.layer_id = 7
                moe2.moe_ep_rank = 0
                self.assertEqual(spill_placement(moe2), ([0, 1, 2, 3], [4, 5]))

    def test_ensure_spill_experts_swaps_then_map_ids(self):
        import torch
        from torch import nn

        from sglang.srt.layers.moe.dsv41_expert_spill import ensure_spill_experts

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        moe = nn.Module()
        moe._dsv41_expert_lru = lru
        ids = torch.tensor([[6, 7, -1]], dtype=torch.int32)
        out = ensure_spill_experts(moe, ids)
        self.assertIs(out, ids)
        self.assertEqual(int(ids[0, 2]), -1)
        self.assertEqual(int(ids[0, 0]), lru.physical_ids([6])[0])
        self.assertEqual(int(ids[0, 1]), lru.physical_ids([7])[0])

    def test_host_map_tracks_lru_swaps(self):
        import torch

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        lru._ensure_device_tables(torch.tensor([0], dtype=torch.int32))
        # Tail cold ids 4,5,6 occupy host rows 0,1,2; kept 0..3 on GPU.
        self.assertEqual(int(lru._host_map_table[6]), 2)
        self.assertEqual(int(lru._map_table[6]), -1)
        self.assertEqual(int(lru._map_table[0]), 0)
        lru.ensure([6])
        self.assertGreaterEqual(int(lru._map_table[6]), 0)
        self.assertEqual(int(lru._host_map_table[6]), -1)
        self.assertEqual(int(lru._map_table[0]), -1)
        self.assertGreaterEqual(int(lru._host_map_table[0]), 0)

    def test_remap_decode_shaped_pages_in_not_lru(self):
        import torch
        from torch import nn
        from unittest.mock import patch

        from sglang.srt.layers.moe.dsv41_expert_spill import (
            remap_dispatch_for_expert_spill,
        )

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        moe = nn.Module()
        moe._dsv41_expert_lru = lru
        moe._dsv41_landing_pool = object()
        ids = torch.tensor([[6, 7, -1]], dtype=torch.int32)
        topk = type("T", (), {})()
        topk.topk_ids = ids
        disp = type("D", (), {})()
        disp.topk_output = topk
        called = {}

        def fake_page(m, t):
            called["page"] = True
            return t

        with patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.spill_landing_slots",
            return_value=6,
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.page_in_spill_experts",
            side_effect=fake_page,
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.ensure_spill_experts"
        ) as ens:
            remap_dispatch_for_expert_spill(moe, disp)
        self.assertTrue(called.get("page"))
        ens.assert_not_called()

    def test_decode_shaped_stays_t2_when_landing_is_six(self):
        import torch

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.dsv41_expert_spill import _decode_shaped_topk

        ids6 = torch.zeros(6, 6, dtype=torch.int32)
        ids2 = torch.zeros(2, 6, dtype=torch.int32)
        with envs.SGLANG_DSV41_SPILL_LANDING.override(6):
            self.assertFalse(_decode_shaped_topk(ids6))
            self.assertTrue(_decode_shaped_topk(ids2))

    def test_remap_t6_verify_is_decode_shaped_not_lru(self):
        """T=6 target-verify must page-in, not prefill ensure()."""
        import torch
        from torch import nn
        from unittest.mock import patch

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.dsv41_expert_spill import (
            _decode_shaped_topk,
            remap_dispatch_for_expert_spill,
        )

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        moe = nn.Module()
        moe._dsv41_expert_lru = lru
        moe._dsv41_landing_pool = object()
        ids = torch.zeros(6, 6, dtype=torch.int32)
        ids[0, 0] = 6
        topk = type("T", (), {})()
        topk.topk_ids = ids
        disp = type("D", (), {})()
        disp.topk_output = topk
        called = {}

        def fake_page(m, t):
            called["page"] = True
            called["shape"] = tuple(t.shape)
            return t

        with envs.SGLANG_DSV41_SPILL_LANDING.override(36), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.page_in_spill_experts",
            side_effect=fake_page,
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.ensure_spill_experts"
        ) as ens:
            self.assertTrue(_decode_shaped_topk(ids))
            remap_dispatch_for_expert_spill(moe, disp)
        self.assertTrue(called.get("page"))
        self.assertEqual(called.get("shape"), (6, 6))
        ens.assert_not_called()

    def test_remap_prefill_uses_lru(self):
        import torch
        from torch import nn
        from unittest.mock import patch

        from sglang.srt.layers.moe.dsv41_expert_spill import (
            remap_dispatch_for_expert_spill,
        )

        w13 = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        lru = RoutedExpertLru(w13.clone(), n_shared=1, n_spilled=3)
        lru.apply_shrink()
        moe = nn.Module()
        moe._dsv41_expert_lru = lru
        moe._dsv41_landing_pool = object()
        ids = torch.zeros(8, 3, dtype=torch.int32)
        ids[0, 0] = 6
        topk = type("T", (), {})()
        topk.topk_ids = ids
        disp = type("D", (), {})()
        disp.topk_output = topk
        with patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.spill_landing_slots",
            return_value=6,
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.page_in_spill_experts"
        ) as page, patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.ensure_spill_experts"
        ) as ens:
            remap_dispatch_for_expert_spill(moe, disp)
        page.assert_not_called()
        ens.assert_called_once()

    def test_lru_swap_rejects_host_gpu_layout_mismatch(self):
        import torch

        gpu = torch.zeros(5, 4608, 2560, dtype=torch.int8)
        host = torch.zeros(3, 320, 9216, dtype=torch.int32)
        lru = RoutedExpertLru(
            gpu,
            n_shared=1,
            n_spilled=3,
            hosts=[host],
            logical_n_experts=8,
            already_shrunk=True,
        )
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            lru.ensure([6])

    def test_cpu_offload_tax_mentions_attention(self):
        tax = cpu_offload_gb_tax()
        self.assertIn("CSA2", tax)
        self.assertIn("--cpu-offload-gb", tax)


class TestSpillNumaFailover(CustomTestCase):
    def test_pin_retries_other_stripe_node(self):
        import mmap
        from unittest.mock import patch

        import torch
        from torch import nn

        from sglang.srt.layers.moe.dsv41_expert_spill import pin_spill_host_numa
        from sglang.srt.mem_cache.dsv41_host_placement import EngramNumaError

        moe = nn.Module()
        moe._dsv41_spill_host = {
            "w13_weight": torch.zeros(32, dtype=torch.uint8),
        }
        seen = []

        def fake_mmap(nbytes, *, node, populate=True):
            seen.append(node)
            if node == 1:
                raise EngramNumaError(
                    "mbind(MPOL_BIND, node=1, 32 bytes) failed errno=5"
                )
            page = mmap.PAGESIZE
            map_bytes = ((nbytes + page - 1) // page) * page
            return mmap.mmap(-1, map_bytes)

        class _Cudart:
            def cudaHostRegister(self, *args):
                return 0

            def cudaHostUnregister(self, *args):
                return 0

        with patch(
            "sglang.srt.layers.moe.dsv41_expert_spill._spill_numa_nodes",
            return_value=[1, 0],
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.mmap_numa_thp",
            side_effect=fake_mmap,
        ), patch(
            "torch.cuda.is_available", return_value=True
        ), patch(
            "torch.cuda.cudart", return_value=_Cudart()
        ):
            used = pin_spill_host_numa(moe, 0)
        self.assertEqual(used, 0)
        self.assertEqual(seen, [1, 0])
        self.assertEqual(moe._dsv41_spill_host_node, 0)


class TestGpuLocalNumaDefault(CustomTestCase):
    """Unset NUMA envs must follow the GPUs' node, not one machine's layout."""

    def _spill_nodes(self, gpu_node, host_nodes):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.dsv41_expert_spill import _spill_numa_nodes
        from sglang.srt.mem_cache.dsv41_host_placement import NumaNodeMem

        nodes = {n: NumaNodeMem(n, GIB, GIB) for n in host_nodes}
        with envs.SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES.override(None), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.gpu_numa_node",
            return_value=gpu_node,
        ), patch(
            "sglang.srt.layers.moe.dsv41_expert_spill.read_numa_nodes",
            return_value=nodes,
        ):
            envs.SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES.clear()
            return _spill_numa_nodes()

    def test_spill_starts_on_gpu_node(self):
        self.assertEqual(self._spill_nodes(1, [0, 1]), [1, 0])
        self.assertEqual(self._spill_nodes(0, [0, 1]), [0, 1])
        self.assertEqual(self._spill_nodes(0, [0]), [0])

    def test_engram_node_follows_gpu_unless_set(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache import dsv41_host_placement as hp

        with envs.SGLANG_DSV41_ENGRAM_NUMA_NODE.override(None), patch.object(
            hp, "gpu_numa_node", return_value=0
        ):
            envs.SGLANG_DSV41_ENGRAM_NUMA_NODE.clear()
            self.assertEqual(hp.engram_numa_node(), 0)
            envs.SGLANG_DSV41_ENGRAM_NUMA_NODE.set(1)
            self.assertEqual(hp.engram_numa_node(), 1)


class TestSpillHostMarlinReady(CustomTestCase):
    def test_checkpoint_host_is_not_packed_even_if_w13_trail_matches(self):
        import torch

        gpu_w13 = torch.empty(34, 320, 8704, dtype=torch.int32)
        gpu_scale = torch.empty(34, 160, 4352, dtype=torch.uint8)
        hosts = {
            "w13_weight": torch.empty(14, 320, 8704, dtype=torch.int32),
            "w13_weight_scale_inv": torch.empty(14, 4352, 160, dtype=torch.uint8),
        }
        self.assertFalse(spill_host_is_marlin_packed(hosts, gpu_w13, gpu_scale))

    def test_packed_host_with_matching_scale_is_ready(self):
        import torch

        gpu_w13 = torch.empty(34, 320, 8704, dtype=torch.int32)
        gpu_scale = torch.empty(34, 160, 4352, dtype=torch.uint8)
        hosts = {
            "w13_weight": torch.empty(14, 320, 8704, dtype=torch.int32),
            "w13_weight_scale": torch.empty(14, 160, 4352, dtype=torch.uint8),
        }
        self.assertTrue(spill_host_is_marlin_packed(hosts, gpu_w13, gpu_scale))

    def test_checkpoint_w13_shape_is_not_packed(self):
        import torch

        gpu_w13 = torch.empty(34, 320, 8704, dtype=torch.int32)
        gpu_scale = torch.empty(34, 160, 4352, dtype=torch.uint8)
        hosts = {
            "w13_weight": torch.empty(14, 4352, 2560, dtype=torch.int8),
            "w13_weight_scale_inv": torch.empty(14, 4352, 160, dtype=torch.uint8),
        }
        self.assertFalse(spill_host_is_marlin_packed(hosts, gpu_w13, gpu_scale))


class TestHugetlbAndSpillPool(CustomTestCase):
    def test_round_to_1g(self):
        self.assertEqual(round_to_huge_pages(1, GIB), GIB)
        self.assertEqual(round_to_huge_pages(int(94.5 * GIB), GIB), 95 * GIB)

    def test_engram_two_layers_round_to_pool(self):
        # Two ~94.5 GiB tables → 95 + 95 = 190 × 1G pages.
        a = round_to_huge_pages(int(94.5 * GIB), GIB)
        b = round_to_huge_pages(int(94.5 * GIB), GIB)
        self.assertEqual((a + b) // GIB, 190)

    def test_one_gib_page_from_node1_pool(self):
        nr, free = read_node_hugepages(1, 1048576)
        if free < 1:
            self.skipTest(f"no free 1G hugepage on node 1 (nr={nr} free={free})")
        mm = fd = None
        try:
            mm, fd, map_bytes = mmap_hugetlb(1, node=1, page_bytes=GIB, shared=False)
            self.assertEqual(map_bytes, GIB)
            mm[0] = 7
            self.assertEqual(mm[0], 7)
            addr = ctypes.addressof((ctypes.c_char * 1).from_buffer(mm))
            rss, thp, hugetlb = smaps_huge_kb(addr)
            self.assertGreater(hugetlb, 0, msg=f"rss={rss} thp={thp} hugetlb={hugetlb}")
        finally:
            if mm is not None:
                mm.close()
            if fd is not None and fd >= 0:
                os.close(fd)
        _, free_after = read_node_hugepages(1, 1048576)
        self.assertEqual(free_after, free)

    def test_spill_thp_does_not_consume_1g_pool(self):
        _, free_before = read_node_hugepages(1, 1048576)
        mm = mmap_numa_thp(4 * 1024 * 1024, node=1)
        try:
            mm[0] = 1
            _, free_after = read_node_hugepages(1, 1048576)
            self.assertEqual(free_after, free_before)
        finally:
            mm.close()


if __name__ == "__main__":
    unittest.main()
