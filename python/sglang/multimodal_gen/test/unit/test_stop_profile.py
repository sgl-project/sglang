import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.control_requests import StopProfileReq


@pytest.mark.parametrize("host_fails", [False, True])
@pytest.mark.parametrize("worker", ["ok", "error", "exception"])
def test_stop_profile_attempts_both_processes(monkeypatch, host_fails, worker):
    profiler = Mock()
    if host_fails:
        profiler.stop.side_effect = RuntimeError("host stop failed")
    forward = AsyncMock(
        return_value=SimpleNamespace(
            error="worker stop failed" if worker == "error" else None
        )
    )
    if worker == "exception":
        forward.side_effect = RuntimeError("worker stop failed")
    state = dict(profiler=profiler, is_profiling=True, profile_id="123")
    monkeypatch.setattr(http_server, "_global_profiler_state", state)
    monkeypatch.setattr(http_server.async_scheduler_client, "forward", forward)
    logger = Mock()
    monkeypatch.setattr(http_server, "logger", logger)

    response = asyncio.run(http_server.stop_profile())

    profiler.stop.assert_called_once_with(export_trace=True, dump_rank=None)
    forward.assert_awaited_once_with(StopProfileReq(export_trace=True))
    assert response.status_code == (500 if host_fails else 200)
    assert (b"host stop failed" in response.body) == host_fails
    assert (b"worker stop failed" in response.body) == (worker != "ok")
    assert logger.info.called == (not host_fails and worker == "ok")
    assert state == dict(profiler=None, is_profiling=False, profile_id=None)
    assert asyncio.run(http_server.stop_profile()).status_code == 400
