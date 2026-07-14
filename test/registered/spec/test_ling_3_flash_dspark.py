import os

os.environ.setdefault("SGLANG_RAGGED_VERIFY_MODE", "static")

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4200, stage="base-c", runner_config="4-gpu-h20")

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

TARGET_MODEL = "/root/models/ling-3.0-flash"
DRAFT_MODEL = "/root/models/ling-3.0-flash-dspark-draft"


class TestLing3FlashDSpark(CustomTestCase):
    """End-to-end DSpark spec decoding coverage on the Ling-3.0-flash target.

    Guards the DSpark worker's hybrid linear-attention (KDA) state commit:
    without ``commit_mamba_states_after_verify`` after accept, the Bailing-MoE-V3
    target diverges from pure-target greedy decoding on multi-step outputs.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TARGET_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.55",
            "--max-running-requests",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                # cuda-graph path mutates server_args.cuda_graph_bs during
                # init_cuda_graphs; the strict config-mutation guard raises on
                # that today. Disable it here to match the manual launch script.
                "SGLANG_STRICT_CONFIG_MUTATION": "0",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=None,
            num_threads=128,
            gsm8k_data_path="/root/datasets/gsm8k/test.jsonl",
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.90)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (ling-3.0-flash dspark) with tp4\n"
                f"score={metrics['score']:.4f}\n"
            )


if __name__ == "__main__":
    unittest.main()
