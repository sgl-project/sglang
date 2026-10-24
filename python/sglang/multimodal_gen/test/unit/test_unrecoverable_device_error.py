# SPDX-License-Identifier: Apache-2.0
"""A forward error that breaks the device must not leave the server serving.

After a non-OOM CUDA driver error (sgl-project/sglang#38167), every later request
either ran 5x slower or failed on a caching-allocator assert, while /health kept
reporting ok. Such an error now rejects later requests and turns /health 503.
"""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    GPUWorker,
    _is_unrecoverable_device_error,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient

# the two errors reported in #38167, verbatim
_DRIVER_ERROR = "CUDA driver error: device not ready"
_ALLOCATOR_ASSERT = (
    '!handles_.at(i) INTERNAL ASSERT FAILED at "/__w/pytorch/pytorch/c10/cuda/'
    'CUDACachingAllocator.cpp":467, please report a bug to PyTorch.'
)


def _wrapped(message: str) -> Exception:
    try:
        try:
            raise RuntimeError(message)
        except RuntimeError as cause:
            raise ValueError("stage failed") from cause
    except ValueError as error:
        return error


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(_DRIVER_ERROR),
        RuntimeError(_ALLOCATOR_ASSERT),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        _wrapped(_DRIVER_ERROR),
    ],
)
def test_device_breaking_errors_are_unrecoverable(error):
    assert _is_unrecoverable_device_error(error)


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory"),
        ValueError("height must be divisible by 16"),
    ],
)
def test_oom_and_request_errors_stay_recoverable(error):
    assert not _is_unrecoverable_device_error(error)


@pytest.fixture
def worker(monkeypatch):
    device = SimpleNamespace(reset_peak_memory_stats=Mock(), empty_cache=Mock())
    monkeypatch.setattr(gpu_worker_module.torch, "get_device_module", lambda: device)
    monkeypatch.setattr(
        type(gpu_worker_module.current_platform), "is_cpu", lambda self: False
    )
    monkeypatch.setattr(
        type(gpu_worker_module.current_platform), "is_mps", lambda self: False
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    worker.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(supports_auto_residency=False)
    )
    worker._realtime_sessions = SimpleNamespace(attach=lambda req: None)
    worker._release_warmup_pool = lambda req: None
    worker._record_output_peak_memory = lambda *args, **kwargs: None
    return worker, device


def _forward_raising(worker, error):
    req = SimpleNamespace(
        request_id="req",
        is_warmup=False,
        extra={},
        metrics=None,
        trace_ctx=MagicMock(),
        log=lambda **kwargs: None,
    )

    def forward():
        raise error

    return worker._execute_forward_common(
        req,
        forward_fn=forward,
        log_reqs=[req],
        return_req=False,
        save_output_paths=lambda output_batch: None,
        error_context="request req",
    )


def test_driver_error_marks_the_worker_and_skips_empty_cache(worker):
    worker, device = worker

    output = _forward_raising(worker, RuntimeError(_DRIVER_ERROR))

    assert output.device_unrecoverable
    assert _DRIVER_ERROR in worker.unrecoverable_device_error
    # empty_cache on a broken allocator can raise out of the error handler
    device.empty_cache.assert_not_called()


def test_oom_keeps_the_worker_serving(worker):
    worker, device = worker

    output = _forward_raising(worker, torch.OutOfMemoryError("CUDA out of memory"))

    assert not output.device_unrecoverable
    assert worker.unrecoverable_device_error is None
    device.empty_cache.assert_called_once_with()


def _scheduler(unrecoverable_device_error):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._async_output_save = False
    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False,
        unrecoverable_device_error=unrecoverable_device_error,
        execute_forward=Mock(return_value=OutputBatch()),
    )
    return scheduler


@pytest.mark.parametrize("num_reqs", [1, 3])
def test_scheduler_rejects_generation_without_forward(num_reqs):
    scheduler = _scheduler(f"request earlier: {_DRIVER_ERROR}")
    reqs = [SimpleNamespace(is_warmup=False) for _ in range(num_reqs)]

    result = scheduler._handle_generation(reqs, allow_dynamic_batching=True)

    outputs = result if isinstance(result, list) else [result]
    assert len(outputs) == num_reqs
    for output in outputs:
        assert output.device_unrecoverable
        assert _DRIVER_ERROR in output.error
        assert "restart the server" in output.error
    scheduler.worker.execute_forward.assert_not_called()


def test_rejected_request_group_gets_one_reply():
    scheduler = _scheduler(f"request earlier: {_DRIVER_ERROR}")
    group = [SimpleNamespace(is_warmup=False) for _ in range(2)]

    result = scheduler._handle_generation(group, allow_dynamic_batching=False)

    assert isinstance(result, OutputBatch)
    assert result.device_unrecoverable


def _client_replying(reply):
    socket = MagicMock()
    socket.send = AsyncMock()
    socket.recv = AsyncMock(return_value=pickle.dumps(reply))
    client = AsyncSchedulerClient()
    client.context = SimpleNamespace(socket=lambda _socket_type: socket)
    client.server_args = SimpleNamespace(
        scheduler_rpc_timeout=None, scheduler_endpoints=["tcp://scheduler"]
    )
    client.request_logger = MagicMock()
    return client


def _health_status(client, monkeypatch):
    monkeypatch.setattr(http_server, "async_scheduler_client", client)
    warmup_done = asyncio.Event()
    warmup_done.set()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(server_warmup_done=warmup_done))
    )
    response = asyncio.run(http_server.health(request))
    return 200 if isinstance(response, dict) else response.status_code


def test_unrecoverable_reply_turns_health_unhealthy(monkeypatch):
    error = f"Error executing request req: {_DRIVER_ERROR}"
    client = _client_replying(OutputBatch(error=error, device_unrecoverable=True))

    asyncio.run(client.forward(object()))

    assert client.unrecoverable_device_error == error
    assert _health_status(client, monkeypatch) == 503


def test_ordinary_request_error_keeps_health_ok(monkeypatch):
    client = _client_replying(OutputBatch(error="height must be divisible by 16"))

    asyncio.run(client.forward(object()))

    assert client.unrecoverable_device_error is None
    assert _health_status(client, monkeypatch) == 200
