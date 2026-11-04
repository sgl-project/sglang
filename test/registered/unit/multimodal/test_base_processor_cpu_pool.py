"""The multimodal CPU preprocess pool defaults to at most 16 workers on many-core hosts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


def _max_workers(cpu_count, env=None):
    stub = SimpleNamespace(mm_feature_transport="shm")
    with (
        patch.dict(os.environ, env or {}, clear=False),
        patch("os.cpu_count", return_value=cpu_count),
        patch("concurrent.futures.ProcessPoolExecutor") as pool,
    ):
        if env is None:
            os.environ.pop("SGLANG_CPU_WORKERS", None)
        BaseMultimodalProcessor._create_cpu_executor(stub)
    return pool.call_args.kwargs["max_workers"]


class TestCpuPoolSize(CustomTestCase):
    def test_capped_on_many_core_host(self):
        self.assertEqual(_max_workers(512), 16)

    def test_small_host_uses_all_cores(self):
        self.assertEqual(_max_workers(4), 4)

    def test_unknown_cpu_count(self):
        self.assertEqual(_max_workers(None), 1)

    def test_env_override(self):
        self.assertEqual(_max_workers(512, {"SGLANG_CPU_WORKERS": "64"}), 64)


if __name__ == "__main__":
    unittest.main()
