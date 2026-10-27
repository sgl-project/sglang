"""The sj-mlfq policy serves correctly: the eval score holds with its gate,
demotion and priority retraction active, under forced retraction and the
strict memory check."""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=330, stage="base-b", runner_config="1-gpu-small")


class TestSjMlfqScheduling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Small chunks so long prompts run as several chunks past the gate;
        # forced retraction exercises the priority retraction order and the
        # level a retracted request keeps.
        with (
            envs.SGLANG_TEST_RETRACT.override(True),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--schedule-policy",
                    "sj-mlfq",
                    "--enable-priority-scheduling",
                    "--retraction-policy",
                    "priority",
                    "--disable-priority-preemption",
                    "--chunked-prefill-size",
                    "512",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=32,
        )

        metrics = run_sgl_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.64)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"


if __name__ == "__main__":
    unittest.main()
