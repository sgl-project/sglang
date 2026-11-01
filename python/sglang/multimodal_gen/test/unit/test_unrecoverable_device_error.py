# SPDX-License-Identifier: Apache-2.0
"""A non-OOM device error (#38167) must exit the worker instead of serving on a
broken device while /health still reports ok."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    GPUWorker,
    _is_unrecoverable_device_error,
)

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
    worker._record_output_peak_memory = Mock()
    return worker, device


def _forward_raising(worker, error, propagate_forward_errors=False):
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
        propagate_forward_errors=propagate_forward_errors,
    )


@pytest.mark.parametrize("propagate", [False, True])
@pytest.mark.parametrize("message", [_DRIVER_ERROR, _ALLOCATOR_ASSERT])
def test_device_error_exits_the_worker(worker, message, propagate):
    worker, device = worker

    # must escape the scheduler's request-level `except Exception` handlers
    with pytest.raises(SystemExit, match="restart the server") as exc_info:
        _forward_raising(worker, RuntimeError(message), propagate)

    assert message in str(exc_info.value.__cause__)
    # empty_cache on a broken allocator can raise out of the error handler
    device.empty_cache.assert_not_called()
    worker._record_output_peak_memory.assert_not_called()


class _Exited(Exception):
    pass


def test_scheduler_process_exits_on_device_error(monkeypatch):
    for name in (
        "initialize_current_platform",
        "kill_itself_when_parent_died",
        "configure_logger",
        "globally_suppress_loggers",
        "_bind_worker_to_gpu_numa_node",
        "set_cuda_arch",
        "set_musa_arch",
        "init_diffusion_tracing",
    ):
        monkeypatch.setattr(gpu_worker_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_worker_module.PortArgs, "from_server_args", lambda server_args: None
    )

    class _BrokenDeviceScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def event_loop(self):
            raise gpu_worker_module._FatalDeviceError(
                "Unrecoverable device error; restart the server."
            )

    monkeypatch.setattr(scheduler_module, "Scheduler", _BrokenDeviceScheduler)

    events = []

    def exit_(code):
        events.append(("exit", code))
        # the real os._exit never returns; raising lets the test continue
        raise _Exited(code)

    monkeypatch.setattr(gpu_worker_module.os, "_exit", exit_)
    monkeypatch.setattr(
        gpu_worker_module.gc, "collect", lambda: events.append(("cleanup",))
    )

    with pytest.raises(_Exited):
        gpu_worker_module.run_scheduler_process(
            local_rank=0, rank=0, server_args=SimpleNamespace(), pipe_writer=Mock()
        )

    # exits at the process boundary before cleanup can touch the broken device
    assert events[0] == ("exit", 1)
