"""The scheduler watchdog dumps the pools once initialization ends; the
invariant checker it reads is attached before that."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    invariant_checker as checker_mod,
)
from sglang.srt.managers.scheduler_components.engine_state import (  # noqa: E402
    EngineFlags,
    LaunchCounters,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerWatchdog(CustomTestCase):
    def test_dump_reads_the_attached_checker_after_initialization(self):
        counters, flags = LaunchCounters(), EngineFlags()
        with patch.object(checker_mod, "WatchdogRaw") as raw:
            watchdog = checker_mod.SchedulerWatchdog(
                counters=counters, flags=flags, watchdog_timeout=1.0
            )
        dump_info = raw.call_args.kwargs["dump_info"]
        self.assertEqual(dump_info(), "")

        watchdog.invariant_checker = SimpleNamespace(
            pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: "stats"),
            _check_all_pools=lambda stats: (False, [f"pools: {stats}"]),
        )
        flags.is_initializing = False
        counters.cur_batch_for_debug = SimpleNamespace(
            batch_size=lambda: 2, reqs=["a", "b"]
        )
        self.assertEqual(
            dump_info(),
            "scheduler.cur_batch_for_debug.batch_size()=2\n"
            "scheduler.cur_batch_for_debug.reqs=['a', 'b']\n"
            "pools: stats",
        )

    def test_checker_is_attached_before_initialization_ends(self):
        source = textwrap.dedent(inspect.getsource(Scheduler.__init__))
        statements = ast.parse(source).body[0].body
        lines = [ast.unparse(statement) for statement in statements]
        checker = lines.index("self.init_invariant_checker()")
        attach = lines.index("self.init_watchdog_diagnostics()")
        done = lines.index("self.flags.is_initializing = False")
        self.assertLess(checker, attach)
        self.assertLess(attach, done)


if __name__ == "__main__":
    unittest.main()
