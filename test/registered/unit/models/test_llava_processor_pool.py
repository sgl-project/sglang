import asyncio
import concurrent.futures
import threading
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import Mock

import pytest

from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BrokenExecutor(concurrent.futures.Executor):
    def submit(self, _fn, /, *_args, **_kwargs):
        raise BrokenProcessPool("worker exited")


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


def test_broken_pool_is_replaced_once_for_concurrent_failures():
    processor = object.__new__(LlavaImageProcessor)
    failed_executor = Mock()
    replacement_executor = Mock()
    processor.cpu_executor = failed_executor
    processor._cpu_executor_lock = threading.Lock()
    processor._create_cpu_executor = Mock(return_value=replacement_executor)

    processor._replace_broken_cpu_executor(failed_executor)
    processor._replace_broken_cpu_executor(failed_executor)

    assert processor.cpu_executor is replacement_executor
    processor._create_cpu_executor.assert_called_once_with()
    failed_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
