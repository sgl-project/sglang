"""Default size of the multimodal CPU preprocess pool."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


class TestCpuPoolSize(CustomTestCase):
    def test_default_worker_count(self):
        """A 512-core host must not fork 512 workers on the first multimodal request."""
        processor = SimpleNamespace(mm_feature_transport="shm")
        for cpu_count, environment, expected_workers in (
            (512, {}, 16),
            (4, {}, 4),
            (None, {}, 1),
            (512, {"SGLANG_CPU_WORKERS": "64"}, 64),
        ):
            with self.subTest(cpu_count=cpu_count, environment=environment):
                with (
                    patch.dict(os.environ, environment),
                    patch("os.cpu_count", return_value=cpu_count),
                    patch("concurrent.futures.ProcessPoolExecutor") as executor_class,
                ):
                    if not environment:
                        os.environ.pop("SGLANG_CPU_WORKERS", None)
                    BaseMultimodalProcessor._create_cpu_executor(processor)
                self.assertEqual(
                    executor_class.call_args.kwargs["max_workers"], expected_workers
                )


if __name__ == "__main__":
    unittest.main()
