"""CPU regressions for OFFLINE prefetch deadlines and BLOCKING compatibility."""

import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "tools/sglang-simulator/src")
)

from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.sglang.cache_controller import C_PrefetchOperationHook
from sglang_simulator.simulation.sglang.unified_radix_cache import (
    C_UnifiedRadixCacheHook,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(
    est_time=5,
    suite="base-a-test-cpu",
    disabled="Run by _pr-test-simulator-cpu.yml with simulator dependencies",
)


class TestPrefetchTimeout(CustomTestCase):
    def setUp(self):
        StateManager.reset()
        self.addCleanup(StateManager.reset)

        class Operation:
            def __init__(self, request_id, *, pages):
                self.request_id = request_id
                self.hash_value = list(range(pages))
                self.start_time = time.monotonic()

        C_PrefetchOperationHook.hook(Operation)
        self.Operation = Operation

    def make_cache(self, mode="OFFLINE", base=1, per_page=0.25):
        native_timeout = Mock(return_value=True)

        class Cache:
            def check_hicache_events(self):
                pass

            def _prefetch_timeout_check_linear_func(self, operation):
                return native_timeout(operation)

        with patch.object(Envs, "simulation_mode", return_value=mode):
            C_UnifiedRadixCacheHook.hook(Cache)
        cache = Cache()
        cache.prefetch_timeout_base = base
        cache.prefetch_timeout_per_page = per_page
        return cache, native_timeout

    def test_bootstrap_stamps_native_unified_prefetch_operations(self):
        # A fresh interpreter also checks registration/import order, as used by
        # spawned scheduler workers. Importing a fake class cannot cover this.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from types import SimpleNamespace; "
                "from sglang_simulator.simulation.sglang.hook_bootstrap "
                "import install_simulator_hooks; "
                "install_simulator_hooks(); "
                "from sglang_simulator.simulation.manager import StateManager; "
                "from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller "
                "import PrefetchOperation; "
                "StateManager.set_global_clock(12.5); "
                "op = PrefetchOperation(SimpleNamespace(rid='request'), []); "
                "assert op.sim_start_time == 12.5; "
                "assert op.request_id == 'request'",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_virtual_deadline_expires_without_wall_time_advancing(self):
        cache, native = self.make_cache()
        StateManager.set_global_clock(10)
        with patch.object(time, "monotonic", return_value=500):
            op = self.Operation("request", pages=4)
            self.assertEqual(op.start_time, 500)
            self.assertEqual(op.sim_start_time, 10)
            StateManager.set_global_clock(12)
            self.assertFalse(cache._prefetch_timeout_check_linear_func(op))
            StateManager.step_global_clock(0.001)
            self.assertTrue(cache._prefetch_timeout_check_linear_func(op))
        native.assert_not_called()

    def test_wall_time_cannot_expire_offline_prefetch(self):
        cache, _ = self.make_cache()
        with patch.object(time, "monotonic", return_value=100):
            op = self.Operation("request", pages=0)
        StateManager.step_global_clock(0.5)
        with patch.object(time, "monotonic", return_value=10000):
            self.assertFalse(cache._prefetch_timeout_check_linear_func(op))

    def test_query_wait_counts_from_operation_creation(self):
        cache, _ = self.make_cache(base=1, per_page=0)
        op = self.Operation("pending-query", pages=0)
        StateManager.step_global_clock(1.5)
        # Discovering hits and allocating pages must not restart the deadline.
        op.hash_value = ["hit"]
        op.host_indices = list(range(256))
        self.assertTrue(cache._prefetch_timeout_check_linear_func(op))

    def test_each_operation_has_its_own_start_time(self):
        cache, _ = self.make_cache(base=1, per_page=0)
        first = self.Operation("first", pages=1)
        StateManager.step_global_clock(0.75)
        second = self.Operation("second", pages=1)
        StateManager.step_global_clock(0.5)
        self.assertTrue(cache._prefetch_timeout_check_linear_func(first))
        self.assertFalse(cache._prefetch_timeout_check_linear_func(second))

    def test_timeout_uses_current_hit_count_and_configuration(self):
        cache, _ = self.make_cache(base=1, per_page=0.25)
        op = self.Operation("request", pages=4)
        StateManager.step_global_clock(1.5)
        self.assertFalse(cache._prefetch_timeout_check_linear_func(op))
        op.hash_value = op.hash_value[:1]
        self.assertTrue(cache._prefetch_timeout_check_linear_func(op))
        cache.prefetch_timeout_base = 2
        self.assertFalse(cache._prefetch_timeout_check_linear_func(op))

    def test_blocking_delegates_to_native_timeout(self):
        cache, native = self.make_cache(mode="BLOCKING")
        op = self.Operation("request", pages=4)
        StateManager.step_global_clock(10000)
        for expected in (False, True):
            native.return_value = expected
            self.assertEqual(cache._prefetch_timeout_check_linear_func(op), expected)
        self.assertEqual(native.call_count, 2)
        native.assert_called_with(op)


if __name__ == "__main__":
    unittest.main()
