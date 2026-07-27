import os
import subprocess
import tempfile
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    CONFIG_YAML_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _create_clean_subprocess_env,
    _wait_for_server_health,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-4-npu-a3",
    nightly=True,
)


class TestConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    @classmethod
    def setUpClass(cls):
        # launch server with "--config" parameter
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        host = parsed_url.hostname
        port = str(parsed_url.port)
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--config",
            CONFIG_YAML_PATH,
            "--host",
            host,
            "--port",
            port,
        ]
        env = _create_clean_subprocess_env(os.environ.copy())
        cls.process = subprocess.Popen(command, stdout=None, stderr=None, env=env)
        _wait_for_server_health(
            cls.process, DEFAULT_URL_FOR_TEST, None, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_config(self):
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


class TestConfigPriority(CustomTestCase):
    """Testcase: Verify set the parameter set in the command line have a higher priority than set in config.yaml,
    set false model path in the command, set right model path in the config.yaml,
    will use false model path service start fail .

    [Test Category] Parameter
    [Test Target] --config
    """

    def test_config_priority(self):
        # will use false model path (/nonexistent/Qwen/Qwen3-32B) service start fail
        error_message = "404 page not found"
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        ) as out_log_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        ) as err_log_file:
            try:
                popen_launch_server(
                    "/nonexistent/Qwen/Qwen3-32B",
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--config", CONFIG_YAML_PATH],
                    return_stdout_stderr=(out_log_file, err_log_file),
                )
            except Exception as e:
                self.assertIn(
                    "Server process exited with code 1.",
                    str(e),
                )
            finally:
                err_log_file.seek(0)
                content = err_log_file.read()
                self.assertIn(error_message, content)


if __name__ == "__main__":
    unittest.main()
