# SPDX-License-Identifier: Apache-2.0
"""A non-OOM device error (#38167) must exit the worker instead of serving on a
broken device while /health still reports ok."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.executors import (
    pipeline_executor as pipeline_executor_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req

# the two errors reported in #38167, verbatim
_DRIVER_ERROR = "CUDA driver error: device not ready"
_ALLOCATOR_ASSERT = (
    '!handles_.at(i) INTERNAL ASSERT FAILED at "/__w/pytorch/pytorch/c10/cuda/'
    'CUDACachingAllocator.cpp":467, please report a bug to PyTorch.'
)


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


def _run_grouped_requests(worker, monkeypatch, error, replica_size):
    """Two grouped requests; the first one's DiT/VAE forward raises ``error``."""
    monkeypatch.setattr(
        gpu_worker_module,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: True, is_mps=lambda: False),
    )
    replica = lambda: SimpleNamespace(world_size=replica_size)
    monkeypatch.setattr(gpu_worker_module, "get_replica_group", replica)
    monkeypatch.setattr(
        pipeline_executor_module, "get_replica_group", replica, raising=False
    )
    executor = object.__new__(SyncExecutor)
    executor.execute_group = lambda stages, batches, server_args: batches

    def execute(stages, batch, server_args):
        if batch.request_id == "first":
            if callable(error):
                raise error()
            raise error
        return OutputBatch()

    executor.execute = execute
    first_stage = SimpleNamespace(
        iter_sequential_requests=lambda batch, server_args: iter((batch,))
    )
    worker.pipeline = SimpleNamespace(
        forward_batch_sequentially=lambda reqs, server_args: (
            executor.execute_group_sequentially([first_stage, object()], reqs, None)
        )
    )
    worker.server_args.pipeline_config.supports_sequential_multi_output_inference = (
        lambda: False
    )
    monkeypatch.setattr(Req, "log", lambda self, **kwargs: None)
    reqs = [
        Req(sampling_params=SamplingParams(prompt="p", seed=1, request_id=request_id))
        for request_id in ("first", "second")
    ]
    return list(worker.execute_forward_sequentially(reqs))


@pytest.mark.parametrize(
    "error, replica_size",
    [
        (RuntimeError(_DRIVER_ERROR), 1),
        (torch.OutOfMemoryError("allocator OOM"), 2),
        (torch.AcceleratorError("CUDA error: out of memory"), 2),
    ],
)
def test_grouped_request_fatal_error_exits_the_worker(
    worker, monkeypatch, error, replica_size
):
    worker, _ = worker

    with pytest.raises(SystemExit, match="restart the server"):
        _run_grouped_requests(worker, monkeypatch, error, replica_size)


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("allocator OOM"),
        torch.AcceleratorError("CUDA error: out of memory"),
    ],
)
def test_grouped_request_oom_on_a_single_rank_fails_only_that_request(
    worker, monkeypatch, error
):
    worker, _ = worker

    outputs = _run_grouped_requests(worker, monkeypatch, error, 1)

    assert str(error) in outputs[0].error
    assert outputs[1].error is None


def test_grouped_request_quoting_a_domain_name_fails_only_that_request(
    worker, monkeypatch
):
    worker, _ = worker

    outputs = _run_grouped_requests(
        worker, monkeypatch, ValueError("domain cuda error: x"), 1
    )

    assert "domain cuda error: x" in outputs[0].error
    assert outputs[1].error is None


def test_grouped_request_wrapped_device_error_exits_the_worker(worker, monkeypatch):
    worker, _ = worker

    def wrapped():
        try:
            raise RuntimeError(_DRIVER_ERROR)
        except RuntimeError as cause:
            raise ValueError("unknown domain") from cause

    with pytest.raises(SystemExit, match="restart the server"):
        _run_grouped_requests(worker, monkeypatch, wrapped, 1)


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
