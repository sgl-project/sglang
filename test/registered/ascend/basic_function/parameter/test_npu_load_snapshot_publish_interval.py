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

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestNpuLoadSnapshotPublishInterval(CustomTestCase):
    """Verify --load-snapshot-publish-interval does not break basic
    server startup, inference, or the /v1/loads endpoint on NPU.

    This parameter controls how often (every N decode iterations) the
    scheduler publishes load snapshots to shared memory, consumed by
    /v1/loads and DP dispatch.  Smaller N = fresher metrics but more
    overhead.

    Note: The throttling counter is internal to ShmLoadSnapshotWriter
    and not exposed via any API.  Prefill always publishes immediately
    (force=True), so an E2E test can verify the endpoint is reachable
    but cannot distinguish N=1 from N=15 in terms of publish frequency.

    [Test Category] Parameter
    [Test Target] --load-snapshot-publish-interval
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    NPU_FIXTURE_ARGS = ["--attention-backend", "ascend"]

    @classmethod
    def _launch_server(cls, extra_args):
        """Launch server with NPU fixture args + extra_args."""
        all_args = cls.NPU_FIXTURE_ARGS + extra_args
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=all_args,
        )

    def test_interval_1_basic_functionality(self):
        """--load-snapshot-publish-interval 1 → server starts normally,
        basic inference works as expected.  Verifies the parameter is
        accepted and does not break core functionality.
        """
        process = self._launch_server(["--load-snapshot-publish-interval", "1"])
        try:
            # Health check
            health = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(health.status_code, 200)

            # Basic inference
            gen_resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
            )
            self.assertEqual(gen_resp.status_code, 200)
            self.assertIn("Paris", gen_resp.text)

            # /v1/loads endpoint is reachable
            loads_resp = requests.get(f"{self.base_url}/v1/loads")
            self.assertEqual(loads_resp.status_code, 200)
            self.assertIn("loads", loads_resp.json())
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
