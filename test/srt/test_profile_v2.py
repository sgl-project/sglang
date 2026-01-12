import os
import shutil
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestStartProfile(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp()
        envs.SGLANG_TORCH_PROFILER_DIR.set(cls.output_dir)
        envs.SGLANG_PROFILE_V2.set(True)
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def setUp(self):
        self._clear_profile_dir()

    def test_profile_by_stage(self):
        self._start_profile(
            profile_by_stage=True,
            num_steps=10,
        )

        self._post_request()

        self._check_profile_output(pattern="*-prefill*", expect_existence=True)
        self._check_profile_output(pattern="*-decode*", expect_existence=True)

    def test_decode_only(self):
        self._start_profile(
            profile_by_stage=True,
            profile_stages=["decode"],
            num_steps=10,
        )

        self._post_request()

        self._check_profile_output(pattern="*-prefill*", expect_existence=False)  # NOTE
        self._check_profile_output(pattern="*-decode*", expect_existence=True)

    def _start_profile(self, **kwargs):
        """Start profiling with optional parameters."""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/start_profile",
            json=kwargs if kwargs else None,
        )
        self.assertEqual(response.status_code, 200)

    def _post_request(self):
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

    def _clear_profile_dir(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

    def _check_profile_output(self, pattern: str, expect_existence: bool):
        self.assertTrue(
            os.path.isdir(self.output_dir), "Output directory does not exist."
        )
        self.assertEqual(
            len(list(Path(self.output_dir).glob(pattern))) > 0,
            expect_existence,
            f"Does not find {pattern=} ({list(Path(self.output_dir).glob('**/*'))=})",
        )


if __name__ == "__main__":
    unittest.main()
