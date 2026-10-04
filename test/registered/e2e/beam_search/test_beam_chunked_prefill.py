"""A beam prompt longer than the chunk budget finishes, whether or not it starts on
a tree prefix, and the server stays up through the strict idle pool check."""

import os
import re
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

CHUNK = 128
SENTENCES = [
    f"Survey {i}: the river carried silt past mile {3 * i + 7} toward the delta."
    for i in range(60)
]
# Several chunks past the tree prefix below.
PROMPT = " ".join(SENTENCES)
TREE_PREFIX = " ".join(SENTENCES[:15])
# Leaves the tree within its first sentence, so its chunks own their KV.
FRESH_PROMPT = " ".join(reversed(SENTENCES))
BEAM_WIDTH = 4
REQUEST_TIMEOUT_S = 60
CHUNKED_PATTERN = re.compile(r"Prefill batch.*#pending-token: [1-9]")
LEAK_MARKER = "memory leak detected"


class TestBeamChunkedPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_path = f"/tmp/beam_chunked_prefill_{os.getpid()}.log"
        cls.log = open(cls.log_path, "w")
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--chunked-prefill-size", str(CHUNK)],
                return_stdout_stderr=(cls.log, cls.log),
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "log"):
            cls.log.close()

    def test_chunked_beam_prompts_finish_and_server_survives_idle(self):
        self.post(TREE_PREFIX, {"max_new_tokens": 1, "temperature": 0})
        on_prefix = self.beam(PROMPT)
        self.beam(FRESH_PROMPT)
        # Each beam request ended with the server idle, where the pool check runs.
        self.post(TREE_PREFIX, {"max_new_tokens": 1, "temperature": 0})

        self.assertGreater(
            on_prefix["meta_info"]["cached_tokens"],
            0,
            "the beam prompt matched no tree prefix",
        )
        with open(self.log_path) as f:
            log = f.read()
        self.assertRegex(
            log, CHUNKED_PATTERN, "no prefill batch was chunked; nothing was tested"
        )
        self.assertNotIn(LEAK_MARKER, log)

    def beam(self, text):
        body = self.post(
            text,
            {"beam_width": BEAM_WIDTH, "n": BEAM_WIDTH, "max_new_tokens": 4},
        )
        self.assertGreater(body["meta_info"]["prompt_tokens"], 3 * CHUNK)
        self.assertEqual(len(body["meta_info"]["beam_results"]), BEAM_WIDTH)
        return body

    def post(self, text, sampling_params):
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={"text": text, "sampling_params": sampling_params},
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.exceptions.Timeout:
            self.fail(
                f"no response in {REQUEST_TIMEOUT_S}s; a chunked beam prefill "
                "that does not resume where the last chunk ended never finishes"
            )
        self.assertEqual(resp.status_code, 200, resp.text[:400])
        return resp.json()


if __name__ == "__main__":
    unittest.main()
