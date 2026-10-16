"""Pause, resume and shutdown under pipeline parallelism: the PP ring launches
nothing while the engine is paused, and leaves its loop on the shutdown
request a SIGTERM sends.

Usage:
python3 -m unittest test_pp_pause_shutdown.TestPPPauseResume
python3 -m unittest test_pp_pause_shutdown.TestPPShutdown
"""

import signal
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="2-gpu-large")

_PP_ARGS = ["--pp-size", 2]


class TestPPPauseResume(PauseResumeInPlaceMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_PP_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestPPShutdown(CustomTestCase):
    def test_sigterm_ends_every_rank(self):
        """Every PP rank's scheduler leaves its loop on the shutdown request,
        so the server does not kill one mid-release."""
        base_url = DEFAULT_URL_FOR_TEST
        with (
            tempfile.TemporaryFile("w+") as stdout,
            tempfile.TemporaryFile("w+") as stderr,
        ):
            process = popen_launch_server(
                DEFAULT_MODEL_NAME_FOR_TEST,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_PP_ARGS,
                return_stdout_stderr=(stdout, stderr),
            )
            try:
                requests.post(
                    base_url + "/generate",
                    json={"text": "Hello", "sampling_params": {"max_new_tokens": 8}},
                    timeout=60,
                ).raise_for_status()
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=120)
            finally:
                kill_process_tree(process.pid)
            # Let the output relay threads drain.
            time.sleep(1)
            stderr.seek(0)
            log = stderr.read()
        self.assertIn("Gracefully exiting", log)
        self.assertNotIn("Schedulers still alive", log)


if __name__ == "__main__":
    unittest.main()
