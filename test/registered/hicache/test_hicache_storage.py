from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=99, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")

import shutil
import tempfile
import time
import unittest

from sglang.srt.utils import is_hip
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

_is_hip = is_hip()


class TestHiCache(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.64
    mmlu_num_examples = 256
    mmlu_num_threads = 32

    # Set before the server launch so tearDownClass can clean up after a
    # setUpClass that raised part-way (CustomTestCase runs it either way).
    process = None
    storage_dir = None

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The file backend defaults to a persistent /tmp/hicache and only evicts
        # when a cap is configured, so an un-redirected run leaks its 8 MiB
        # pages onto the runner for good.
        cls.storage_dir = tempfile.mkdtemp(prefix="hicache_storage_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.7,
                "--hicache-size",
                100 if not _is_hip else 200,
                "--page-size",
                "64",
                "--hicache-storage-backend",
                "file",
            ],
            env={"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir},
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        time.sleep(5)
        if cls.storage_dir is not None:
            shutil.rmtree(cls.storage_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
