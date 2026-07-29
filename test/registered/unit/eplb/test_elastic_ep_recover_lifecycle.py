"""Lightweight lifecycle tests for Elastic-EP fault → recover flow.

This test suite is a **CPU-only, single-process substitute** for the multi-GPU
end-to-end recovery test in `test/manual/ep/test_elastic_recover.py`. It is
designed to validate the exact code path modified by the P0.1 "skip
host-device sync in forward fast path" PR, but without requiring a Mooncake
cluster, 8 × GPUs, or a 271 GB DeepSeek model.

Why this is meaningful:

The end-to-end test's real signal is a **state-machine invariant**:

    forward()  =>  maybe_recover_ep_ranks()  =>  {fast-path | slow-path}

    where the transition between fast/slow paths is driven **only** by
    `ElasticEPState.active_ranks_cpu`, which must stay in lock-step with
    `active_ranks`.

If the mirror contract holds and `maybe_recover_ep_ranks` reads only the
mirror on the fast path, the P0.1 change is semantically equivalent to the
old dual-check. This file pins exactly that contract by constructing a
**real** `ElasticEPState` (not a mock), driving it through the same
transitions the multi-GPU test provokes, and asserting the return values
and side-effects of `maybe_recover_ep_ranks` match the expected lifecycle.

Reference:
- `python/sglang/srt/elastic_ep/elastic_ep.py::maybe_recover_ep_ranks`
- `python/sglang/srt/elastic_ep/elastic_ep.py::ElasticEPState`
- `test/manual/ep/test_elastic_recover.py::TestElasticRecover4To4`
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.elastic_ep import elastic_ep as elastic_ep_module
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    maybe_recover_ep_ranks,
)
from sglang.test.test_utils import CustomTestCase


def _make_real_state(world_size: int = 8) -> ElasticEPState:
    """Build a real ElasticEPState on CPU (bypassing distributed init).

    Mirrors `ElasticEPStateManager._build_state` but pins device=cpu so we
    don't need `torch.distributed` initialized. This is the same state
    shape a healthy 8-rank cluster would produce right after boot.
    """
    active = torch.ones(world_size, dtype=torch.int32, device="cpu")
    state = ElasticEPState(
        active_ranks=active,
        last_active_ranks=active.clone(),
        active_ranks_cpu=active.detach().cpu().clone(),
        effective_ep_size=world_size,
        original_ep_size=world_size,
    )
    return state


def _make_tp_group_from_state(state: ElasticEPState) -> SimpleNamespace:
    """Bind a state's tensors to a stand-in tp_group with the two fields
    that `maybe_recover_ep_ranks` reads."""
    return SimpleNamespace(
        active_ranks=state.active_ranks,
        active_ranks_cpu=state.active_ranks_cpu,
    )


class TestElasticEpRecoverLifecycle(CustomTestCase):
    """Full fault → degraded → recover cycle, mirroring the E2E test phases."""

    WORLD_SIZE = 8
    LOCAL_EP_SIZE = 4  # matches TestElasticRecover4To4's LOCAL_EP_SIZE

    def setUp(self):
        # Fresh state each test — the E2E test also starts from a clean cluster.
        self.state = _make_real_state(self.WORLD_SIZE)
        # Install the state into the singleton so `maybe_recover_ep_ranks`'s
        # slow-path call to `ElasticEPStateManager.instance().reset()` targets
        # our test state rather than a stale global.
        self._prev_instance = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = self.state

    def tearDown(self):
        ElasticEPStateManager._instance = self._prev_instance

    # ------------------------------------------------------------------
    # Helpers that mirror the manual test's `_generate_ok` semantics but
    # target the actual code path we care about.
    # ------------------------------------------------------------------

    def _call_recover(self, try_recover_ranks_result: bool = False):
        """Invoke maybe_recover_ep_ranks with stubbed collectives."""
        tp_group = _make_tp_group_from_state(self.state)
        with (
            patch.object(
                elastic_ep_module,
                "try_recover_ranks",
                return_value=try_recover_ranks_result,
            ) as m_try_recover,
            patch.object(
                elastic_ep_module, "broadcast_global_expert_location_metadata"
            ) as m_broadcast,
            patch.object(
                elastic_ep_module,
                "get_healthy_expert_location_src_rank",
                return_value=0,
            ),
        ):
            eplb_manager = MagicMock()
            result = maybe_recover_ep_ranks(
                tp_group=tp_group,
                eplb_manager=eplb_manager,
                model_config=MagicMock(),
                moe_ep_rank=0,
            )
        return SimpleNamespace(
            result=result,
            try_recover_ranks=m_try_recover,
            broadcast=m_broadcast,
            eplb_manager=eplb_manager,
        )

    def _inject_rank_faults(self, faulty_ranks):
        """Simulate `maybe_rebalance_after_rank_fault`'s bookkeeping.

        In the E2E test, killing the joiner node causes the scheduler to
        eventually call `maybe_rebalance_after_rank_fault`, which:
          - flips the faulty entries in `active_ranks` to 0
          - calls `snapshot_active_to_last` (which internally refreshes
            the CPU mirror via `sync_active_to_cpu`)

        We reproduce the same side-effects here.
        """
        for r in faulty_ranks:
            self.state.active_ranks[r] = 0
        self.state.snapshot_active_to_last()

    # ------------------------------------------------------------------
    # Phase 1 — initial healthy service (mirrors: _generate_ok("initial service"))
    # ------------------------------------------------------------------

    def test_phase1_healthy_cluster_takes_fast_path(self):
        """No faults => every forward returns immediately via the fast path.

        The E2E test calls `/generate` right after boot and expects HTTP 200.
        Here the equivalent signal is `result is False` (no recover triggered)
        AND that the slow-path collective `try_recover_ranks` was never called.
        """
        outcome = self._call_recover()
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_not_called()
        outcome.broadcast.assert_not_called()

    def test_phase1_repeated_forwards_never_touch_slow_path(self):
        """Fast path must be idempotent under repeated invocation.

        Real forward loops call this every step; if fast-path had any hidden
        state mutation, repeated calls would break the invariant.
        """
        for _ in range(50):
            outcome = self._call_recover()
            self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_not_called()

    # ------------------------------------------------------------------
    # Phase 2 — degraded service (mirrors: kill joiner, then _generate_ok)
    # ------------------------------------------------------------------

    def test_phase2_after_fault_slow_path_engaged(self):
        """One or more ranks faulted => slow path fires with the right list.

        In the E2E test, we kill node1 (ranks 4..7). Here we inject the same
        fault set and assert `try_recover_ranks` is called with those exact
        indices. This is the moment where the P0.1 change would break if the
        CPU mirror were not authoritative — the fast-path check must correctly
        *fail closed* and delegate to slow path.
        """
        faulty = list(range(self.LOCAL_EP_SIZE, self.WORLD_SIZE))  # [4,5,6,7]
        self._inject_rank_faults(faulty)

        # try_recover_ranks returns False => peer not ready yet (recover pending)
        outcome = self._call_recover(try_recover_ranks_result=False)
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_called_once_with(faulty)
        outcome.broadcast.assert_not_called()
        outcome.eplb_manager.reset_generator.assert_not_called()

    def test_phase2_degraded_service_polls_recover_repeatedly(self):
        """While peer isn't ready yet, every forward re-attempts recovery.

        The E2E test's `RECOVER_WAIT_SECONDS` + repeated `_generate_ok`
        represents this polling window: fast-path stays disengaged, slow-path
        keeps calling `try_recover_ranks(faulty)` until it eventually returns
        True. We simulate 5 poll cycles here.
        """
        faulty = [4, 5, 6, 7]
        self._inject_rank_faults(faulty)

        for _ in range(5):
            outcome = self._call_recover(try_recover_ranks_result=False)
            self.assertFalse(outcome.result)
            outcome.try_recover_ranks.assert_called_once_with(faulty)

    def test_phase2_partial_fault_only_reports_missing_ranks(self):
        """A subset of ranks down => `ranks_to_recover` == those subset indices.

        Verifies the slow-path's derivation from `active_ranks_cpu` (via the
        AND of both tensors) is correct even for asymmetric fault patterns.
        """
        # A more scattered fault pattern than the E2E's contiguous node-kill.
        self._inject_rank_faults([1, 3, 6])

        outcome = self._call_recover(try_recover_ranks_result=False)
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_called_once_with([1, 3, 6])

    # ------------------------------------------------------------------
    # Phase 3 — recovery completes (mirrors: recover_joiner rejoins, wait for done)
    # ------------------------------------------------------------------

    def test_phase3_successful_recover_returns_true_and_resets_state(self):
        """`try_recover_ranks` returns True => full recover sequence fires.

        This is the E2E test's "recover ranks [4,5,6,7] done" moment:
          - broadcast_global_expert_location_metadata is invoked
          - eplb_manager.reset_generator is called
          - ElasticEPStateManager.instance().reset() runs
          - maybe_recover_ep_ranks returns True
        """
        faulty = [4, 5, 6, 7]
        self._inject_rank_faults(faulty)

        outcome = self._call_recover(try_recover_ranks_result=True)

        self.assertTrue(outcome.result)
        outcome.try_recover_ranks.assert_called_once_with(faulty)
        outcome.broadcast.assert_called_once()
        outcome.eplb_manager.reset_generator.assert_called_once()

        # After a successful recover, the state's active_ranks and its CPU
        # mirror must both be back to the healthy pattern (all 1s for the
        # effective ep size). This is exactly what `ElasticEPState.reset`
        # guarantees, and it is what allows the next forward to fall back
        # into the fast path.
        self.assertTrue(bool(self.state.active_ranks.all()))
        self.assertTrue(bool(self.state.active_ranks_cpu.all()))

    def test_phase3_post_recover_forward_returns_to_fast_path(self):
        """After a successful recover, subsequent forwards must take fast path.

        This is the *core* semantic the E2E test's post-recovery `_generate_ok`
        requests validate. If the CPU mirror were not correctly refreshed by
        `state.reset()`, the fast path would still see zeros and we'd loop in
        the slow path forever.
        """
        # 1) Fault
        self._inject_rank_faults([4, 5, 6, 7])
        # 2) Successful recover
        self._call_recover(try_recover_ranks_result=True)
        # 3) Rebind tp_group to the now-reset state tensors (state.reset()
        #    zeroed and re-populated `active_ranks`, cloned to `active_ranks_cpu`).
        outcome = self._call_recover()
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_not_called()

    # ------------------------------------------------------------------
    # Invariant checks — the core contract the P0.1 optimization relies on.
    # ------------------------------------------------------------------

    def test_invariant_cpu_mirror_matches_gpu_after_every_mutation(self):
        """After any state mutation, CPU mirror must equal the source tensor.

        This is the mirror contract that P0.1 relies on. Rather than proving
        it globally (that requires reading every mutation site), we check it
        holds at each transition the recovery lifecycle produces.
        """
        # Post-boot
        self.assertTrue(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu)
        )
        # Post-fault
        self._inject_rank_faults([2, 5])
        self.assertTrue(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu)
        )
        # Post-recover
        self._call_recover(try_recover_ranks_result=True)
        self.assertTrue(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu)
        )

    def test_invariant_stale_cpu_mirror_would_be_detected(self):
        """Regression guard: if a future refactor writes to `active_ranks`
        without going through `snapshot_active_to_last()`, this test locks
        the failure mode into a deterministic signal.

        We artificially skip the publish call (simulating a broken mutation
        site) and assert that `maybe_recover_ep_ranks` would incorrectly
        stay on the fast path. This documents *why* every mutation site
        must conclude with `snapshot_active_to_last()` — the fast-path
        optimization is only safe under that invariant.
        """
        # Directly mutate active_ranks WITHOUT the sanctioned publish call.
        self.state.active_ranks[3] = 0
        # (deliberately DON'T call snapshot_active_to_last)
        self.assertFalse(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu),
            "Test precondition: mirror should now be stale.",
        )

        outcome = self._call_recover()
        # With the P0.1 change, the fast path trusts the (stale) CPU mirror
        # and returns False. This is documented behavior: any mutation to
        # `active_ranks` MUST be paired with `snapshot_active_to_last()`.
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_not_called()

        # Now do the publish that the mutation site *should* have done, and
        # verify the slow path immediately kicks in on the next call.
        self.state.snapshot_active_to_last()
        outcome = self._call_recover(try_recover_ranks_result=False)
        self.assertFalse(outcome.result)
        outcome.try_recover_ranks.assert_called_once_with([3])

    def test_snapshot_active_to_last_also_refreshes_cpu_mirror(self):
        """Contract check: `snapshot_active_to_last` is the single publish
        point and MUST refresh `active_ranks_cpu` as part of its job.

        This test pins the coupling introduced in the P0.1 PR: previously
        callers had to invoke both `snapshot_active_to_last()` and
        `sync_active_to_cpu()` in sequence. Merging them into a single
        `snapshot_active_to_last()` call removes an entire class of
        "forgot to sync" bugs. Any future refactor that separates these
        two operations will break this test and must be rejected.
        """
        # Mutate active_ranks in isolation from the CPU mirror.
        self.state.active_ranks[1] = 0
        self.state.active_ranks[4] = 0
        # Precondition: CPU mirror is now stale.
        self.assertFalse(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu)
        )

        # Single publish call must simultaneously:
        #   (a) record the new baseline in last_active_ranks
        #   (b) refresh the CPU mirror
        self.state.snapshot_active_to_last()

        # (a) baseline updated
        self.assertTrue(
            torch.equal(self.state.active_ranks, self.state.last_active_ranks),
            "snapshot_active_to_last must record active_ranks into last_active_ranks",
        )
        # (b) CPU mirror refreshed as a side-effect — this is the key
        # coupling that lets the forward fast path skip a host-device sync.
        self.assertTrue(
            torch.equal(self.state.active_ranks, self.state.active_ranks_cpu),
            "snapshot_active_to_last must also refresh active_ranks_cpu",
        )


class TestElasticEpRecoverLifecycleAlternateShapes(CustomTestCase):
    """Cover non-8-rank shapes that the E2E test never exercises."""

    def test_2_rank_cluster_recover_cycle(self):
        state = _make_real_state(world_size=2)
        prev = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = state
        try:
            # Fault rank 1
            state.active_ranks[1] = 0
            state.snapshot_active_to_last()

            tp_group = _make_tp_group_from_state(state)
            with (
                patch.object(
                    elastic_ep_module, "try_recover_ranks", return_value=True
                ) as m_try_recover,
                patch.object(
                    elastic_ep_module, "broadcast_global_expert_location_metadata"
                ),
                patch.object(
                    elastic_ep_module,
                    "get_healthy_expert_location_src_rank",
                    return_value=0,
                ),
            ):
                result = maybe_recover_ep_ranks(
                    tp_group=tp_group,
                    eplb_manager=MagicMock(),
                    model_config=MagicMock(),
                    moe_ep_rank=0,
                )
            self.assertTrue(result)
            m_try_recover.assert_called_once_with([1])
        finally:
            ElasticEPStateManager._instance = prev

    def test_16_rank_cluster_recover_cycle(self):
        """The E2E test is 8-rank; assert the code is shape-agnostic."""
        state = _make_real_state(world_size=16)
        prev = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = state
        try:
            # Fault a full node's worth of ranks in a 16-rank cluster.
            faulty = list(range(8, 16))
            for r in faulty:
                state.active_ranks[r] = 0
            state.snapshot_active_to_last()

            tp_group = _make_tp_group_from_state(state)
            with (
                patch.object(
                    elastic_ep_module, "try_recover_ranks", return_value=True
                ) as m_try_recover,
                patch.object(
                    elastic_ep_module, "broadcast_global_expert_location_metadata"
                ),
                patch.object(
                    elastic_ep_module,
                    "get_healthy_expert_location_src_rank",
                    return_value=0,
                ),
            ):
                result = maybe_recover_ep_ranks(
                    tp_group=tp_group,
                    eplb_manager=MagicMock(),
                    model_config=MagicMock(),
                    moe_ep_rank=0,
                )
            self.assertTrue(result)
            m_try_recover.assert_called_once_with(faulty)
        finally:
            ElasticEPStateManager._instance = prev


if __name__ == "__main__":
    unittest.main()
