# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def worker(monkeypatch):
    releases = []
    clock = _Clock()
    monkeypatch.setattr(gpu_worker_module.time, "monotonic", clock)
    monkeypatch.setattr(
        gpu_worker_module.torch,
        "get_device_module",
        lambda: SimpleNamespace(empty_cache=lambda: releases.append(clock.now)),
    )
    monkeypatch.setattr(
        type(gpu_worker_module.current_platform), "is_cpu", lambda self: False
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker.defer_cache_release = False
    worker._cache_release_due = None
    worker.is_output_rank = True
    worker._materialize_output_transport = lambda *args: None
    worker._record_output_peak_memory = lambda *args, **kwargs: None
    return worker, releases, clock


def _finish_request(worker, *, deferred=False):
    req = SimpleNamespace(
        request_id="request",
        perf_dump_path=None,
        is_warmup=False,
        suppress_logs=True,
        return_raw_frames=False,
    )
    worker._finalize_output_batch(
        output_batch=OutputBatch(),
        req=req,
        save_output_paths=lambda batch: None,
        output_metrics=[],
        deferred=deferred,
    )


@pytest.mark.parametrize("deferred", [False, True])
def test_cache_is_released_only_after_the_scheduler_idles(worker, deferred):
    worker, releases, clock = worker
    worker.defer_cache_release = True

    _finish_request(worker, deferred=deferred)
    worker.release_cache_if_idle()
    assert releases == []

    # a back-to-back request finishing before the delay pushes it out again
    clock.now += gpu_worker_module._IDLE_CACHE_RELEASE_S / 2
    _finish_request(worker, deferred=deferred)
    clock.now += gpu_worker_module._IDLE_CACHE_RELEASE_S / 2
    worker.release_cache_if_idle()
    assert releases == []

    clock.now += gpu_worker_module._IDLE_CACHE_RELEASE_S
    worker.release_cache_if_idle()
    worker.release_cache_if_idle()
    assert releases == [clock.now]


def test_without_an_idle_hook_each_request_still_releases(worker):
    worker, releases, clock = worker

    _finish_request(worker)
    assert releases == [clock.now]
    _finish_request(worker, deferred=True)
    assert releases == [clock.now]
    worker.release_cache_if_idle()
    assert releases == [clock.now]
