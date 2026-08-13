"""Unit tests for the load-snapshot source adapters."""

from __future__ import annotations

import sys

import pytest

from sglang.srt.load_reporter.snapshot_source import ManagerLoadSnapshotSource
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestManagerLoadSnapshotSource:
    @pytest.mark.asyncio
    async def test_manager_source_can_read_without_entering_manager_event_loop(self):
        expected_loads = [object()]

        class Reader:
            def read_all(self):
                return expected_loads

        class Manager:
            async def get_loads(self, include):
                raise AssertionError("background reporter entered manager event loop")

        reader = Reader()
        source = ManagerLoadSnapshotSource(Manager(), {0}, snapshot_reader=reader)

        assert await source.get_loads() is expected_loads

    def test_manager_source_tracks_elastic_worker_count(self):
        class Manager:
            elastic_worker_count = 1

        manager = Manager()
        source = ManagerLoadSnapshotSource(manager, {0})

        assert source.expected_dp_ranks() == frozenset({0})

        manager.elastic_worker_count = 3
        assert source.expected_dp_ranks() == frozenset({0, 1, 2})

        manager.elastic_worker_count = 2
        assert source.expected_dp_ranks() == frozenset({0, 1})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
