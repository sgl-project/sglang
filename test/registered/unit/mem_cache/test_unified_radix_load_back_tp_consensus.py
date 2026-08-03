"""HiCache load-back must reach the same verdict on every TP rank.

`UnifiedRadixCache._load_back_transfers` sets how much of a request's prefix
counts as cached. Deciding that from rank-local pool state lets ranks disagree,
and they then enter the per-layer TP all_reduce with differently-sized tensors
and deadlock. See `_load_back_transfers` for the full rationale; these tests
pin the two properties that follow from it.

The second property is the subtle one: the load-back path must stay free of
collectives. A rank rejected by the KV-budget gates in
`PrefillAdder.add_one_req` also breaks out of the admission loop, so unequal
load-back counts imply an already divergent batch -- but a collective here
would turn that into an immediate scheduler-thread hang, and it would be wrong
outright for any future caller whose reach is genuinely rank-local.

    python -m pytest \
      test/registered/unit/mem_cache/test_unified_radix_load_back_tp_consensus.py -v
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictResult
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

PAGE_SIZE = 256
KV_TOKENS = 4 * PAGE_SIZE
# Distinct sentinels so a mis-indexed unpack cannot look like a correct one.
WRITE_ACKS = 7
LOAD_ACKS = 11
STORAGE_SIZES = (21, 22, 23, 24)


def _make_stub_cache(
    avail, evictable, *, evicted=None, load_fails=False, enable_storage=False
):
    """A UnifiedRadixCache stubbed down to the load-back decision path.

    The capacity snapshot comes from the production initializer rather than
    being hand-set, so the stub cannot drift from what `__init__` actually
    builds.
    """
    cache = object.__new__(UnifiedRadixCache)
    UnifiedRadixCache._init_group_capacity(cache)

    cache.load_back_threshold = 10
    cache.ongoing_load_back = {}
    cache.sidecar_pool_specs = []
    cache.metrics_collector = None
    cache.pp_rank = 0

    host_indices = torch.arange(KV_TOKENS, dtype=torch.int64)
    kv_xfer = SimpleNamespace(host_indices=host_indices)
    # tree_core must exist first: enable_storage is a property that writes
    # through to it.
    cache.tree_core = SimpleNamespace(
        build_load_back_spec=MagicMock(return_value=(kv_xfer, {})),
        commit_load_back=MagicMock(return_value=[]),
    )
    cache.enable_storage = enable_storage
    cache._build_sidecar_transfers = MagicMock(return_value=[])
    cache._apply_cache_actions = MagicMock()

    cache.supports_swa = MagicMock(return_value=True)
    cache.token_to_kv_pool_allocator = SimpleNamespace(
        full_available_size=MagicMock(return_value=avail),
        available_size=MagicMock(return_value=avail),
    )
    cache.evictable_size = MagicMock(return_value=evictable)

    # By default evict() reports back exactly what it was asked for; `evicted`
    # forces an under-delivery, which the decision path must not branch on.
    cache.evict = MagicMock(
        side_effect=lambda params: EvictResult(
            num_tokens_evicted=params.num_tokens if evicted is None else evicted
        )
    )
    cache.dec_lock_ref = MagicMock()
    cache.dec_host_lock_ref = MagicMock()
    cache.inc_lock_ref = MagicMock(
        return_value=SimpleNamespace(to_dec_params=MagicMock(return_value=None))
    )
    cache.cache_controller = SimpleNamespace(
        load=MagicMock(
            return_value=(
                None if load_fails else torch.arange(KV_TOKENS, dtype=torch.int64)
            )
        ),
        ack_write_queue=[],
        ack_load_queue=[],
        extra_host_mem_release_queues={},
        prefetch_revoke_queue=SimpleNamespace(qsize=lambda: STORAGE_SIZES[0]),
        prefetch_hit_queue=SimpleNamespace(qsize=lambda: STORAGE_SIZES[1]),
        ack_backup_queue=SimpleNamespace(qsize=lambda: STORAGE_SIZES[2]),
        host_mem_release_queue=SimpleNamespace(qsize=lambda: STORAGE_SIZES[3]),
    )
    cache._count_ready_acks = MagicMock(
        side_effect=lambda q: (
            WRITE_ACKS if q is cache.cache_controller.ack_write_queue else LOAD_ACKS
        )
    )
    return cache


def _drive(cache):
    return cache._load_back_transfers(
        node_id=1,
        mem_quota=None,
        req=None,
        result=SimpleNamespace(delta=0),
        ancestor_lock_params=None,
        host_anchor_params=None,
    )


class TestLoadBackDecisionFromSnapshot(CustomTestCase):
    """Holding the tree and quota fixed, the verdict depends only on the group
    snapshot -- never on this rank's own pool."""

    def _seeded(self, avail, evictable, group_avail, group_evictable, **kw):
        cache = _make_stub_cache(avail, evictable, **kw)
        cache._set_group_capacity(group_avail, group_evictable, avail)
        return cache

    def test_divergent_local_pools_do_not_change_the_verdict(self):
        flush = self._seeded(
            avail=99 * PAGE_SIZE,
            evictable=0,
            group_avail=2 * PAGE_SIZE,
            group_evictable=8 * PAGE_SIZE,
        )
        tight = self._seeded(
            avail=2 * PAGE_SIZE,
            evictable=8 * PAGE_SIZE,
            group_avail=2 * PAGE_SIZE,
            group_evictable=8 * PAGE_SIZE,
        )
        # Pinned explicitly rather than only compared: under rank-local logic
        # the flush rank would not evict at all.
        self.assertTrue(_drive(flush))
        self.assertTrue(_drive(tight))
        for cache in (flush, tight):
            self.assertEqual(
                cache.evict.call_args[0][0].num_tokens, KV_TOKENS - 2 * PAGE_SIZE
            )

    def test_group_without_room_vetoes(self):
        cache = self._seeded(
            avail=8 * PAGE_SIZE,
            evictable=8 * PAGE_SIZE,
            group_avail=PAGE_SIZE,
            group_evictable=PAGE_SIZE,
        )
        cache.metrics_collector = MagicMock()
        self.assertFalse(_drive(cache))
        cache.cache_controller.load.assert_not_called()
        cache.metrics_collector.increment_load_back_group_veto.assert_called_once()

    def test_ample_group_room_skips_eviction(self):
        cache = self._seeded(
            avail=PAGE_SIZE,
            evictable=0,
            group_avail=8 * PAGE_SIZE,
            group_evictable=0,
        )
        cache.metrics_collector = MagicMock()
        self.assertTrue(_drive(cache))
        cache.evict.assert_not_called()
        cache.metrics_collector.increment_load_back_group_veto.assert_not_called()

    def test_budget_is_spent_so_one_step_cannot_double_allocate(self):
        cache = self._seeded(
            avail=0,
            evictable=0,
            group_avail=KV_TOKENS + PAGE_SIZE,
            group_evictable=0,
        )
        self.assertTrue(_drive(cache))
        # One page of headroom left and nothing evictable, so the second load
        # must be vetoed rather than handed the same capacity again.
        self.assertFalse(_drive(cache))

    def test_budget_is_spent_even_when_the_allocation_fails(self):
        """Both exits must move the snapshot identically.

        Refunding only on success would let one rank's allocation failure
        desync the snapshot itself -- the very thing the snapshot prevents.
        """
        ok = self._seeded(
            avail=0, evictable=0, group_avail=KV_TOKENS + PAGE_SIZE, group_evictable=0
        )
        failed = self._seeded(
            avail=0,
            evictable=0,
            group_avail=KV_TOKENS + PAGE_SIZE,
            group_evictable=0,
            load_fails=True,
        )
        failed.metrics_collector = MagicMock()
        self.assertTrue(_drive(ok))
        self.assertFalse(_drive(failed))
        self.assertEqual(ok._group_avail, failed._group_avail)
        failed.metrics_collector.increment_load_back_alloc_failed.assert_called_once()

    def test_under_delivering_eviction_does_not_change_the_decision(self):
        """The measured eviction amount is rank-local; branching on it deadlocks."""
        full = self._seeded(
            avail=0,
            evictable=0,
            group_avail=2 * PAGE_SIZE,
            group_evictable=8 * PAGE_SIZE,
        )
        short = self._seeded(
            avail=0,
            evictable=0,
            group_avail=2 * PAGE_SIZE,
            group_evictable=8 * PAGE_SIZE,
            evicted=0,
        )
        self.assertEqual(_drive(full), _drive(short))
        self.assertEqual(full._group_avail, short._group_avail)
        self.assertEqual(full._group_evictable, short._group_evictable)

    def test_load_back_skipped_before_the_first_group_sync(self):
        cache = _make_stub_cache(avail=99 * PAGE_SIZE, evictable=0)
        cache.metrics_collector = MagicMock()
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="ERROR"
        ) as logs:
            self.assertFalse(_drive(cache))
        self.assertTrue(any("group capacity sync" in m for m in logs.output))
        cache.cache_controller.load.assert_not_called()
        # A missing snapshot is not a veto; conflating them would hide it.
        cache.metrics_collector.increment_load_back_group_veto.assert_not_called()

    def test_capacity_skew_uses_the_pre_reduce_reading(self):
        cache = _make_stub_cache(avail=6 * PAGE_SIZE, evictable=0)
        cache.metrics_collector = MagicMock()
        # A second sample would read the later value; the skew must not.
        cache.token_to_kv_pool_allocator.full_available_size.side_effect = [
            999 * PAGE_SIZE
        ]
        cache._set_group_capacity(2 * PAGE_SIZE, 0, 6 * PAGE_SIZE)
        cache.metrics_collector.set_hicache_group_capacity_skew.assert_called_once_with(
            4 * PAGE_SIZE
        )

    def test_group_above_local_is_reported_not_clamped(self):
        """MIN including this rank cannot exceed it; zero would read as healthy."""
        cache = _make_stub_cache(avail=2 * PAGE_SIZE, evictable=0)
        cache.metrics_collector = MagicMock()
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="ERROR"
        ) as logs:
            cache._set_group_capacity(9 * PAGE_SIZE, 0, 2 * PAGE_SIZE)
        self.assertTrue(any("did not include this rank" in m for m in logs.output))
        cache.metrics_collector.set_hicache_group_capacity_skew.assert_not_called()

    def test_later_pipeline_stage_reports_overgrant_instead_of_skew(self):
        cache = _make_stub_cache(avail=2 * PAGE_SIZE, evictable=0)
        cache.metrics_collector = MagicMock()
        cache.pp_rank = 1
        cache._set_group_capacity(9 * PAGE_SIZE, 0, 2 * PAGE_SIZE)
        cache.metrics_collector.set_hicache_group_capacity_skew.assert_not_called()
        cache.metrics_collector.set_hicache_pp_capacity_overgrant.assert_called_once_with(
            7 * PAGE_SIZE
        )
        # The snapshot itself must still be applied.
        self.assertEqual(cache._group_avail, 9 * PAGE_SIZE)

    def test_failure_paths_report_once_per_step_not_once_per_request(self):
        """Both conditions persist across steps; per-request logging would flood."""
        cache = self._seeded(
            avail=0,
            evictable=0,
            group_avail=99 * KV_TOKENS,
            group_evictable=0,
            load_fails=True,
        )
        for _ in range(5):
            self.assertFalse(_drive(cache))
        self.assertEqual(cache._alloc_failures_this_step, 5)
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="ERROR"
        ) as logs:
            cache._set_group_capacity(99 * KV_TOKENS, 0, 99 * KV_TOKENS)
        self.assertEqual(len([m for m in logs.output if "allocation failed" in m]), 1)
        self.assertTrue(any("5 time(s)" in m for m in logs.output))
        self.assertEqual(cache._alloc_failures_this_step, 0)

    def test_evict_shortfall_is_counted_and_reported(self):
        cache = self._seeded(
            avail=0,
            evictable=0,
            group_avail=2 * PAGE_SIZE,
            group_evictable=8 * PAGE_SIZE,
            evicted=0,
        )
        cache.metrics_collector = MagicMock()
        self.assertTrue(_drive(cache))
        shortfall = KV_TOKENS - 2 * PAGE_SIZE
        counter = cache.metrics_collector.increment_load_back_evict_shortfall_num_tokens
        counter.assert_called_once_with(shortfall)
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
        ) as logs:
            cache._set_group_capacity(2 * PAGE_SIZE, 0, 2 * PAGE_SIZE)
        self.assertTrue(any("under-delivered" in m for m in logs.output))

    def test_reset_rearms_the_missing_snapshot_warning(self):
        """A flush starts a new epoch; the second occurrence must not be silent."""
        cache = _make_stub_cache(avail=0, evictable=0)
        cache._set_group_capacity(PAGE_SIZE, 0, PAGE_SIZE)
        cache._warned_missing_group_capacity = True
        # _reset_full tears down a lot of tree state; only its epoch handling
        # is under test, so let the rest be absorbed.
        cache.tree_core = MagicMock()
        cache.session = MagicMock()
        cache.session_refs = MagicMock()
        cache.cache_controller = None

        cache._reset_full()

        self.assertIsNone(cache._group_avail)
        self.assertIsNone(cache._group_evictable)
        self.assertFalse(cache._warned_missing_group_capacity)

    def test_no_branch_reaches_a_collective(self):
        """The guard the gloo test cannot give: every branch, collective-free.

        Covers the statements of _load_back_transfers itself on every branch,
        which the multi-process test cannot -- it only ever takes one. Callees
        are stubbed, so a collective added inside evict() or the cache
        controller is out of scope here.
        """
        # ample room / veto / evict / alloc-failure, each with its verdict
        cases = [
            (dict(group_avail=8 * PAGE_SIZE, group_evictable=0), True),
            (dict(group_avail=PAGE_SIZE, group_evictable=PAGE_SIZE), False),
            (dict(group_avail=2 * PAGE_SIZE, group_evictable=8 * PAGE_SIZE), True),
            (
                dict(group_avail=8 * PAGE_SIZE, group_evictable=0, load_fails=True),
                False,
            ),
        ]
        boom = MagicMock(side_effect=AssertionError("collective on the load-back path"))
        collectives = ("all_reduce", "barrier", "recv", "isend", "broadcast")

        def guarded(cache):
            cache._all_reduce = boom
            patches = [patch.object(torch.distributed, n, boom) for n in collectives]
            for p in patches:
                p.start()
            try:
                return _drive(cache)
            finally:
                for p in patches:
                    p.stop()

        for kw, expected in cases:
            with self.subTest(**kw):
                # Pinning the verdict stops a threshold change from silently
                # collapsing every case into the same early return.
                cache = self._seeded(avail=0, evictable=0, **kw)
                self.assertEqual(guarded(cache), expected)
        # And the no-snapshot branch, which takes none of the above.
        self.assertFalse(guarded(_make_stub_cache(avail=0, evictable=0)))


# --------------------------------------------------------------------------
# Real gloo process group
# --------------------------------------------------------------------------


def _gloo_rank_main(
    rank, store_path, avails, evictables, load_back_ranks, out_dir, enable_storage
):
    """Entry point for each spawned rank. Must stay module-level (picklable)."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        world_size=2,
        rank=rank,
    )
    try:
        cache = _make_stub_cache(
            avails[rank], evictables[rank], enable_storage=enable_storage
        )
        cache.attn_cp_group = None
        cache.attn_tp_group = None
        cache.tp_group = None  # default (world) group
        cache.tp_world_size = 2
        cache.pp_rank = 0
        cache.pp_size = 1
        cache.pp_group = None
        cache.work_list = []

        # The per-step merged reduce every rank reaches unconditionally.
        counts = cache._sync_hicache_ready_counts()

        verdict = None
        if rank in load_back_ranks:
            verdict = _drive(cache)
        result = {
            "verdict": verdict,
            "group_avail": cache._group_avail,
            "group_evictable": cache._group_evictable,
            "write_acks": counts[0],
            "load_acks": counts[1],
            "storage_sizes": list(counts[2]),
            "evict_calls": [c[0][0].num_tokens for c in cache.evict.call_args_list],
        }
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
            json.dump(result, fh)
    finally:
        torch.distributed.destroy_process_group()


def _run_gloo(
    avails, evictables, load_back_ranks=(0, 1), timeout=120, enable_storage=False
):
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as out_dir:
        # A file store avoids the bind-then-reconnect race of picking a port.
        store_path = os.path.join(out_dir, "store")
        ctx = mp.start_processes(
            _gloo_rank_main,
            args=(
                store_path,
                avails,
                evictables,
                load_back_ranks,
                out_dir,
                enable_storage,
            ),
            nprocs=2,
            join=False,
            daemon=True,
            start_method="spawn",
        )
        try:
            # ProcessContext.join() returns as soon as ANY rank exits, reporting
            # False while others remain, so it has to be driven to completion.
            deadline = time.monotonic() + timeout
            while not ctx.join(timeout=max(0.0, deadline - time.monotonic())):
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        f"gloo ranks did not finish within {timeout}s -- deadlocked"
                    )
            results = []
            for rank in range(2):
                path = os.path.join(out_dir, f"rank{rank}.json")
                if not os.path.exists(path):
                    raise AssertionError(f"rank {rank} produced no result")
                with open(path) as fh:
                    results.append(json.load(fh))
            return results
        finally:
            # Reap before the TemporaryDirectory is torn out from under them,
            # so a real failure is not masked by an rmtree error.
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in ctx.processes:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo required",
)
class TestLoadBackTpConsensusGloo(CustomTestCase):
    def test_storage_queue_sizes_land_after_the_capacity_and_ack_slots(self):
        """With storage on, the variable tail must start at exactly index 4."""
        results = _run_gloo(
            avails=[8 * PAGE_SIZE, 2 * PAGE_SIZE],
            evictables=[PAGE_SIZE, 5 * PAGE_SIZE],
            enable_storage=True,
        )
        for r in results:
            self.assertEqual(r["write_acks"], WRITE_ACKS)
            self.assertEqual(r["load_acks"], LOAD_ACKS)
            self.assertEqual(r["storage_sizes"], list(STORAGE_SIZES))

    def test_merged_reduce_agrees_on_capacity_and_keeps_ack_layout(self):
        """Capacity rides along without displacing the ack counts."""
        results = _run_gloo(
            avails=[8 * PAGE_SIZE, 2 * PAGE_SIZE],
            evictables=[PAGE_SIZE, 5 * PAGE_SIZE],
        )
        self.assertEqual([r["group_avail"] for r in results], [2 * PAGE_SIZE] * 2)
        self.assertEqual([r["group_evictable"] for r in results], [PAGE_SIZE] * 2)
        # The two minima came from different ranks, which is what makes
        # group_avail + group_evictable a safe lower bound for the veto.
        for r in results:
            self.assertEqual(r["write_acks"], WRITE_ACKS)
            self.assertEqual(r["load_acks"], LOAD_ACKS)
            self.assertEqual(r["storage_sizes"], [])

    def test_divergent_availability_yields_one_shared_verdict(self):
        results = _run_gloo(
            avails=[8 * PAGE_SIZE, 2 * PAGE_SIZE],
            evictables=[8 * PAGE_SIZE, 8 * PAGE_SIZE],
        )
        self.assertEqual([r["verdict"] for r in results], [True, True])
        self.assertEqual(
            [r["evict_calls"] for r in results],
            [[KV_TOKENS - 2 * PAGE_SIZE]] * 2,
        )

    def test_group_veto_is_unanimous(self):
        results = _run_gloo(
            avails=[8 * PAGE_SIZE, PAGE_SIZE],
            evictables=[8 * PAGE_SIZE, 0],
        )
        self.assertEqual([r["verdict"] for r in results], [False, False])

    def test_one_sided_load_back_does_not_hang(self):
        """The discriminating case: only one rank reaches load-back.

        The KV-budget gates in PrefillAdder.add_one_req can return NO_TOKEN on a
        tight rank while its peer proceeds to init_load_back, so a collective
        inside the load-back path would block here forever.
        """
        results = _run_gloo(
            avails=[8 * PAGE_SIZE, 2 * PAGE_SIZE],
            evictables=[8 * PAGE_SIZE, 8 * PAGE_SIZE],
            load_back_ranks=(0,),
        )
        self.assertEqual(results[0]["verdict"], True)
        self.assertIsNone(results[1]["verdict"])


if __name__ == "__main__":
    unittest.main()
