import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class _BaseTestDynamicEPLB(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "mooncake",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                # TODO pr-chain: enable later
                # "--enable-expert-distribution-metrics",
                # TODO auto determine these flags
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
                *cls.extra_args,
            ],
            env={
                "SGL_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestDynamicEPLBSimple(_BaseTestDynamicEPLB):
    pass


if __name__ == "__main__":
    unittest.main()
