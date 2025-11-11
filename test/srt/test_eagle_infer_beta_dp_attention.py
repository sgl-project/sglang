import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DP_ATTENTION_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_DP_ATTENTION_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEagleDPAttnServerBase(CustomTestCase, MatchedStopMixin):
    max_running_requests = 64
    attention_backend = "triton"
    spec_steps = 6
    spec_topk = 10
    spec_draft_tokens = 32
    page_size = 1
    other_launch_args = []
    model = DEFAULT_EAGLE_DP_ATTENTION_TARGET_MODEL_FOR_TEST
    draft_model = DEFAULT_EAGLE_DP_ATTENTION_DRAFT_MODEL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model",
            cls.draft_model,
            "--speculative-num-steps",
            cls.spec_steps,
            "--speculative-eagle-topk",
            cls.spec_topk,
            "--speculative-num-draft-tokens",
            cls.spec_draft_tokens,
            "--page-size",
            str(cls.page_size),
            "--tp-size",
            2,
            "--dp-size",
            2,
            "--enable-dp-attention",
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleLargeBS -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.91
        )


if __name__ == "__main__":
    unittest.main()
