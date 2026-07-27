import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(
        self,
        snapshots: list[SimpleNamespace],
        *,
        gracefully_exit: bool = False,
        server_status: ServerStatus = ServerStatus.Up,
        error: Exception | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.gracefully_exit = gracefully_exit
        self.server_status = server_status
        self.elastic_worker_count = 2
        self.error = error
        self.get_loads_calls = 0

    async def get_loads(
        self,
        include: list[str] | None = None,
    ) -> list[SimpleNamespace]:
        self.get_loads_calls += 1
        if include != ["core"]:
            raise AssertionError(f"unexpected load fields: {include}")
        if self.error is not None:
            raise self.error
        return self.snapshots


def _snapshot(dp_rank: int, timestamp: float) -> SimpleNamespace:
    return SimpleNamespace(dp_rank=dp_rank, timestamp=timestamp)


class TestSchedulerHealth(unittest.IsolatedAsyncioTestCase):
    async def _status(self, manager: _FakeTokenizerManager | None) -> int:
        global_state = (
            None if manager is None else SimpleNamespace(tokenizer_manager=manager)
        )
        with patch.object(http_server, "_global_state", global_state):
            return (await http_server.health_scheduler()).status_code

    async def test_healthy_when_all_dp_snapshots_are_fresh(self) -> None:
        now = time.time()
        manager = _FakeTokenizerManager(
            [_snapshot(0, now), _snapshot(1, now)],
        )
        self.assertEqual(await self._status(manager), 200)
        self.assertEqual(manager.get_loads_calls, 1)

    async def test_fails_closed_for_bad_snapshot_sets(self) -> None:
        now = time.time()
        cases = {
            "missing rank": [_snapshot(0, now)],
            "duplicate rank": [_snapshot(0, now), _snapshot(0, now)],
            "stale timestamp": [
                _snapshot(
                    0,
                    now - http_server.SCHEDULER_HEALTH_MAX_STALENESS_SECONDS - 1,
                ),
                _snapshot(1, now),
            ],
            "future timestamp": [
                _snapshot(
                    0,
                    now + http_server.SCHEDULER_HEALTH_MAX_FUTURE_SKEW_SECONDS + 1,
                ),
                _snapshot(1, now),
            ],
            "invalid timestamp": [_snapshot(0, 0), _snapshot(1, now)],
        }
        for name, snapshots in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    await self._status(_FakeTokenizerManager(snapshots)),
                    503,
                )

    async def test_fails_closed_when_state_is_unavailable(self) -> None:
        now = time.time()
        cases = {
            "load read error": _FakeTokenizerManager(
                [_snapshot(0, now), _snapshot(1, now)],
                error=RuntimeError("synthetic load snapshot failure"),
            ),
            "starting": _FakeTokenizerManager(
                [_snapshot(0, now), _snapshot(1, now)],
                server_status=ServerStatus.Starting,
            ),
            "shutting down": _FakeTokenizerManager(
                [_snapshot(0, now), _snapshot(1, now)],
                gracefully_exit=True,
            ),
        }
        for name, manager in cases.items():
            with self.subTest(name=name):
                self.assertEqual(await self._status(manager), 503)

        self.assertEqual(await self._status(None), 503)

    def test_route_is_registered_once(self) -> None:
        routes = [
            route
            for route in http_server.app.routes
            if getattr(route, "path", None) == "/health_scheduler"
        ]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].methods, {"GET"})


if __name__ == "__main__":
    unittest.main()
