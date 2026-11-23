"""One-NPU smoke test for sleep, resume and weight reload without CPU backup."""

import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    generation_signature,
    npu_rl_env,
    npu_server,
    post,
    release_memory,
    resume_memory,
)
from sglang.test.test_utils import CustomTestCase

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
MEMORY_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "512",
    "--max-running-requests",
    "4",
    "--enable-memory-saver",
)


class TestNpuSleepUpdateNoBackup(CustomTestCase):
    def _update_weights_from_disk(self):
        response = post(
            self.base_url, "/update_weights_from_disk", {"model_path": MODEL}
        ).json()
        self.assertIs(
            response.get("success"), True, f"Weight reload failed: {response}"
        )

    def test_sleep_update_matches_no_sleep_reference(self):
        """Check a full weight reload cycle on hardware without CPU backup."""
        self._update_weights_from_disk()
        reference = generation_signature(self.base_url, PROMPTS)

        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights", "kv_cache"])
        self._update_weights_from_disk()
        actual = generation_signature(self.base_url, PROMPTS)

        self.assertEqual(
            actual, reference, "sleep -> resume -> update changed the output"
        )

    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=MEMORY_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
