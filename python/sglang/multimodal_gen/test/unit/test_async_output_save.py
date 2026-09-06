# SPDX-License-Identifier: Apache-2.0

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import _sample_to_uint8_frames
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler, _DeferredOutput
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


def _req():
    return Req(sampling_params=SamplingParams(prompt="test"))


@pytest.mark.parametrize("fail", [False, True])
def test_finalize_selects_worker_device_and_restores_thread_device(monkeypatch, fail):
    current_device = 0
    stream = Mock()

    @contextmanager
    def device(index):
        nonlocal current_device
        previous = current_device
        current_device = index
        try:
            yield
        finally:
            current_device = previous

    def make_stream():
        assert current_device == 1
        return stream

    @contextmanager
    def use_stream(selected):
        assert selected is stream
        assert current_device == 1
        yield

    device_module = SimpleNamespace(
        device=device, Stream=Mock(side_effect=make_stream), stream=use_stream
    )
    monkeypatch.setattr(torch, "get_device_module", lambda: device_module)
    worker = GPUWorker.__new__(GPUWorker)
    worker.local_rank = 1
    worker._deferred_save_stream = None
    done_event = object()

    def finalize(**kwargs):
        assert current_device == 1
        stream.wait_event.assert_called_with(done_event)
        if fail:
            raise OSError("save failed")

    worker._finalize_output_batch = finalize
    for _ in range(2):
        try:
            worker._finalize_deferred(
                output_batch=OutputBatch(),
                req=_req(),
                save_output_paths=lambda _: None,
                output_metrics=[],
                done_event=done_event,
            )
        except OSError:
            assert fail
        else:
            assert not fail
        assert current_device == 0
    device_module.Stream.assert_called_once()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_finalize_materializes_pixels_on_nonzero_worker_device():
    worker = GPUWorker.__new__(GPUWorker)
    worker.local_rank = 1
    worker._deferred_save_stream = None
    observed = []

    def finalize(**kwargs):
        observed.append(
            (
                torch.cuda.current_device(),
                torch.cuda.current_stream().device,
                _sample_to_uint8_frames(kwargs["output_batch"].output[0]),
            )
        )

    worker._finalize_output_batch = finalize

    def run_finalize(output, done_event):
        torch.cuda.set_device(0)
        worker._finalize_deferred(
            output_batch=OutputBatch(output=output),
            req=_req(),
            save_output_paths=lambda _: None,
            output_metrics=[],
            done_event=done_event,
        )
        assert torch.cuda.current_device() == 0

    with torch.cuda.device(1), ThreadPoolExecutor(max_workers=1) as executor:
        producer = torch.cuda.Stream()
        for value in (0.0, 1.0):
            with torch.cuda.stream(producer):
                output = torch.full((1, 3, 1, 2, 2), value, device="cuda:1")
                done_event = torch.cuda.Event()
                done_event.record()
            executor.submit(run_finalize, output, done_event).result(timeout=30)
            device, stream_device, frames = observed[-1]
            assert device == 1
            assert stream_device == torch.device("cuda:1")
            np.testing.assert_array_equal(frames[0], np.full((2, 2, 3), value * 255))


@pytest.mark.parametrize("grouped", [False, True])
def test_generation_waits_for_capacity_before_forward(grouped):
    waiting = Future()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiting.set_result(None)
            return super().result(timeout)

    first, second = ObservedFuture(), Future()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._async_output_save = True
    scheduler._inflight_finalizes = deque([first, second])
    scheduler._ready_replies = SimpleQueue()
    scheduler._return_item_result = Mock()
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False
        )
    )
    reqs = [_req() for _ in range(2 if grouped else 1)]
    completed_output = OutputBatch()

    def forward(*args, **kwargs):
        assert first.done()
        assert not second.done()
        scheduler._return_item_result.assert_called_once_with(
            (b"first", reqs[0]), completed_output
        )
        return OutputBatch()

    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False,
        execute_forward=Mock(side_effect=forward),
        take_deferred_finalize=lambda: lambda: None,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        dispatch = executor.submit(
            scheduler._handle_generation, reqs, allow_dynamic_batching=False
        )
        try:
            done, _ = wait([waiting, dispatch], timeout=10, return_when=FIRST_COMPLETED)
            assert waiting in done
            assert not dispatch.done()
            scheduler.worker.execute_forward.assert_not_called()
            scheduler._ready_replies.put(((b"first", reqs[0]), completed_output))
            first.set_result(None)
            result = dispatch.result(timeout=10)
            assert isinstance(result, OutputBatch if grouped else _DeferredOutput)
            scheduler.worker.execute_forward.assert_called_once()
        finally:
            if not first.done():
                first.set_result(None)
            second.set_result(None)


def test_generation_with_one_pending_finalize_still_overlaps():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._async_output_save = True
    pending = Future()
    scheduler._inflight_finalizes = deque([pending])
    scheduler._ready_replies = SimpleQueue()
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False
        )
    )
    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False,
        execute_forward=Mock(return_value=OutputBatch()),
        take_deferred_finalize=lambda: lambda: None,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        dispatch = executor.submit(scheduler._handle_generation, [_req()])
        try:
            assert isinstance(dispatch.result(timeout=10), _DeferredOutput)
            assert not pending.done()
            scheduler.worker.execute_forward.assert_called_once()
        finally:
            pending.set_result(None)
