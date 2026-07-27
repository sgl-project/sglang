import json
import os
import shutil
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuMsprobeDumpConfig(CustomTestCase):
    """Testcase: verify --msprobe-dump-config enables msProbe, disables
    cuda graph and warmup, and produces dump files in eager mode

    [Test Category] Parameter
    [Test Target] --msprobe-dump-config
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls._dump_dir = tempfile.mkdtemp(prefix="msprobe_dump_")

        cls._tmp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(
            {
                "task": "statistics",
                "level": "L0",
                "step": [0],
                "dump_path": cls._dump_dir,
            },
            cls._tmp_config,
        )
        cls._tmp_config.close()

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--msprobe-dump-config",
                cls._tmp_config.name,
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        os.unlink(cls._tmp_config.name)
        shutil.rmtree(cls._dump_dir, ignore_errors=True)

    def test_msprobe_dump_config_eager_mode(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        self.err_log_file.seek(0)
        err_log = self.err_log_file.read()
        self.assertIn(
            "When msProbe is enabled",
            err_log,
            "Expected stderr to contain 'When msProbe is enabled', proving "
            "--msprobe-dump-config was parsed and cuda graph + warmup were disabled",
        )
        self.assertNotIn(
            "Please install msprobe",
            err_log,
            "Expected stderr NOT to contain 'Please install msprobe', proving "
            "mindstudio-probe is installed and PrecisionDebugger was created",
        )

        # msprobe writes dump.json into per-step subdirectories
        # (e.g. step31/dump.json), not at the root of the dump dir.
        dump_files = []
        for root, _dirs, files in os.walk(self._dump_dir):
            for f in files:
                if f == "dump.json":
                    dump_files.append(os.path.join(root, f))
        self.assertTrue(
            len(dump_files) > 0 and any(os.path.getsize(df) > 0 for df in dump_files),
            f"Expected at least one non-empty dump.json under {self._dump_dir} "
            f"to exist and be non-empty, proving msProbe actually dumped tensor "
            f"statistics during inference. Found: {dump_files}",
        )


if __name__ == "__main__":
    unittest.main()
