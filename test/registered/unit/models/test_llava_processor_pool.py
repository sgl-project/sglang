import asyncio
import concurrent.futures
import os
import threading
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import Mock

import pytest

from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class _BrokenExecutor(concurrent.futures.Executor):
    def submit(self, _fn, /, *_args, **_kwargs):
        raise BrokenProcessPool("worker exited")


class _BlockingExecutor(concurrent.futures.Executor):
    def __init__(self):
        self.release = threading.Event()

    def submit(self, _fn, /, *_args, **_kwargs):
        self.release.wait(timeout=5)
        return concurrent.futures.Future()


def _exit_worker_process():
    os._exit(1)


def test_llava_replaces_broken_pool_without_replaying_request():
    processor = object.__new__(LlavaImageProcessor)
    processor.cpu_executor = _BrokenExecutor()
    processor._processor = Mock()
    processor._replace_broken_cpu_executor = Mock()

    with pytest.raises(BrokenProcessPool, match="worker exited"):
        asyncio.run(processor._process_single_image(b"image", "pad", None))

    processor._replace_broken_cpu_executor.assert_called_once_with(
        processor.cpu_executor
    )


def test_llava_times_out_blocked_pool_submission_without_freezing_loop(monkeypatch):
    monkeypatch.setenv("REQUEST_TIMEOUT", "1")
    processor = object.__new__(LlavaImageProcessor)
    processor.cpu_executor = _BlockingExecutor()
    processor._processor = Mock()
    processor._replace_broken_cpu_executor = Mock()

    async def run_test():
        heartbeat = asyncio.Event()

        async def keep_loop_responsive():
            await asyncio.sleep(0.01)
            heartbeat.set()

        heartbeat_task = asyncio.create_task(keep_loop_responsive())
        try:
            with pytest.raises(asyncio.TimeoutError):
                await processor._process_single_image(b"image", "pad", None)
            assert heartbeat.is_set()
        finally:
            processor.cpu_executor.release.set()
            await heartbeat_task

    asyncio.run(run_test())
    processor._replace_broken_cpu_executor.assert_called_once_with(
        processor.cpu_executor
    )


def test_broken_pool_is_replaced_once_for_concurrent_failures():
    processor = object.__new__(LlavaImageProcessor)
    failed_executor = Mock()
    replacement_executor = Mock()
    processor.cpu_executor = failed_executor
    processor._cpu_executor_lock = threading.Lock()
    processor._create_cpu_executor = Mock(return_value=replacement_executor)
    shutdown_called = threading.Event()
    failed_executor.shutdown.side_effect = lambda **_kwargs: shutdown_called.set()

    processor._replace_broken_cpu_executor(failed_executor)
    processor._replace_broken_cpu_executor(failed_executor)

    assert processor.cpu_executor is replacement_executor
    processor._create_cpu_executor.assert_called_once_with()
    assert shutdown_called.wait(timeout=5)
    failed_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


def test_replacement_pool_runs_after_real_worker_exit(monkeypatch):
    monkeypatch.setenv("SGLANG_CPU_WORKERS", "1")
    processor = object.__new__(LlavaImageProcessor)
    processor.mm_feature_transport = "cpu"
    processor._cpu_executor_lock = threading.Lock()
    failed_executor = processor._create_cpu_executor()
    processor.cpu_executor = failed_executor

    try:
        with pytest.raises(BrokenProcessPool):
            failed_executor.submit(_exit_worker_process).result(timeout=5)
        processor._replace_broken_cpu_executor(failed_executor)
        assert processor.cpu_executor.submit(abs, -1).result(timeout=5) == 1
    finally:
        processor.cpu_executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
