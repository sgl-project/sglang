import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.runtime_context import get_context
from sglang.test.scheduler_request_lists import make_request_registry

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(is_retracted: bool, holds_kv: bool = False) -> MagicMock:
    req = MagicMock()
    req.is_retracted = is_retracted
    # Retracted and never-scheduled requests hold no request row.
    req.kv.holds_kv = holds_kv
    return req


class TestSchedulerFlushCacheAfterRetract(unittest.TestCase):
    """Regression coverage for flush_cache() no-oping while the engine is paused.

    With generation paused and nothing decoding, nothing in waiting_queue
    holds KV/pool state: neither requests pause_generation(mode="retract") just
    moved there, nor requests that were never scheduled at all. The latter are
    always present under high concurrency (in-flight targets exceed engine
    parallelism), where gating the flush on them starves weight updates until
    the caller's retry budget times out (reproduced in RL training at 512
    in-flight requests)."""

    def _new_planner(self) -> StepPlanner:
        enter_scope(self, published_topology())
        planner = StepPlanner.__new__(StepPlanner)
        planner.counters = LaunchCounters()
        planner.flags = EngineFlags()
        planner.core_lists = CoreLists()
        planner.flags._engine_paused = True
        get_context().override("test", disable_overlap_schedule=True)
        planner.core_lists.chunked_req = None
        planner.core_lists.decoding = []
        planner.core_lists.waiting_queue = []
        planner.grammar_manager = MagicMock()
        planner.grammar_manager.grammar_queue = []
        get_context().override("test", enable_hierarchical_cache=False)
        get_context().override("test", enable_lmcache=False)
        get_context().override("test", enable_hisparse=False)
        planner.tree_cache = MagicMock()
        planner.req_to_token_pool = MagicMock()
        planner.token_to_kv_pool_allocator = MagicMock()
        planner.draft_worker = None
        planner.metrics_sink = MagicMock()
        planner.request_registry = make_request_registry(
            core_lists=planner.core_lists,
            grammar_manager=planner.grammar_manager,
            kv=SimpleNamespace(tree_cache=planner.tree_cache),
        )
        return planner

    def test_flush_cache_succeeds_when_only_retracted_reqs_are_waiting(self):
        planner = self._new_planner()
        planner.core_lists.waiting_queue = [_make_req(is_retracted=True)]

        self.assertTrue(planner.flush_cache())
        planner.tree_cache.reset.assert_called_once()
        planner.req_to_token_pool.clear.assert_called_once()
        planner.token_to_kv_pool_allocator.clear.assert_called_once()

    def test_flush_cache_succeeds_with_never_scheduled_reqs_while_paused(self):
        """A request that was queued but never scheduled holds no KV/pool state
        either; a paused-engine flush must not starve on it."""
        planner = self._new_planner()
        planner.core_lists.waiting_queue = [_make_req(is_retracted=False)]

        self.assertTrue(planner.flush_cache())
        planner.tree_cache.reset.assert_called_once()

    def test_paused_flush_refuses_waiting_req_holding_kv(self):
        """A PD-decode request whose KV has arrived waits holding its request
        row; a paused flush would free KV it still owns."""
        planner = self._new_planner()
        planner.core_lists.waiting_queue = [
            _make_req(is_retracted=False),
            _make_req(is_retracted=False, holds_kv=True),
        ]

        self.assertFalse(planner.flush_cache())
        planner.tree_cache.reset.assert_not_called()

    def test_is_fully_idle_default_ignores_nothing(self):
        """Regression guard: every other is_fully_idle() caller passes no args,
        so a non-empty waiting_queue must still report not-idle by default."""
        planner = self._new_planner()
        planner.core_lists.waiting_queue = [_make_req(is_retracted=True)]

        self.assertFalse(planner.request_registry.is_fully_idle())
        self.assertTrue(planner.request_registry.is_fully_idle(ignore_waiting=True))

    def test_flush_cache_does_not_ignore_waiting_reqs_while_unpaused(self):
        """ignore_waiting is gated on _engine_paused inside flush_cache — a
        flush issued without pausing first must not be silently permissive."""
        planner = self._new_planner()
        planner.flags._engine_paused = False
        planner.core_lists.waiting_queue = [_make_req(is_retracted=True)]

        self.assertFalse(planner.flush_cache())
        planner.tree_cache.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
