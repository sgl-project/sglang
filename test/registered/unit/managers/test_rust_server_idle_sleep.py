"""The idle loop of a scheduler that embeds the Rust server parks on its request
ring only with --sleep-on-idle, as Python's parks on its sockets. A server that
holds drained requests back still parks the loop on those without the flag."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _HoldingRustServer(RustServer):
    """Parks on a drained request it holds back, else on the ring."""

    def __init__(self):
        super().__init__(Mock(), http_port=0)
        self.holding = False
        self.parked_on_held = 0

    def wait_request(self, timeout_ms: int) -> None:
        if self.holding:
            self.parked_on_held += 1
            return
        super().wait_request(timeout_ms)

    def install_sigterm_drain(self, notify=None, hold_drain_end=False) -> None:
        # The real one installs a process-wide SIGTERM handler on the server's
        # wakeup fd; draining is not what this test checks.
        pass


class TestRustServerIdleSleep(unittest.TestCase):
    def test_parks_on_held_requests_and_on_the_ring_only_with_sleep_on_idle(self):
        for sleep_on_idle in (False, True):
            with self.subTest(sleep_on_idle=sleep_on_idle):
                rust_server = _HoldingRustServer()
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.get_rust_server_class = lambda: SimpleNamespace(
                    gather_server_endpoints=lambda _: [],
                    gather_frontend_cpu_share=lambda _: (0, 1),
                    launch=lambda _scheduler, frontend_cpu_share=None: rust_server,
                )
                scheduler._engine_paused = False
                scheduler.idle_sleeper = None
                with (
                    get_context().override_server_args(sleep_on_idle=sleep_on_idle),
                    envs.SGLANG_RUST_SERVER.override(True),
                    get_parallel().override(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
                ):
                    scheduler.maybe_init_rust_server(
                        SimpleNamespace(instance_id="idle-sleep-test")
                    )
                    rust_server.holding = True
                    scheduler.maybe_sleep_on_idle()
                    self.assertEqual(rust_server.parked_on_held, 1)

                    rust_server.holding = False
                    scheduler.maybe_sleep_on_idle()
                self.assertEqual(rust_server.server.wait_request.called, sleep_on_idle)


if __name__ == "__main__":
    unittest.main()
