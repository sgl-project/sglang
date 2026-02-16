import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=283, suite="stage-b-test-small-1-gpu")


class TestEagleServerBase(CustomTestCase, MatchedStopMixin):
    max_running_requests = 64
    attention_backend = "triton"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 1
    other_launch_args = []
    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE",
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
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
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
        assert self.process.poll() is None

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1000,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestEagleLargeBS -- {metrics=}")
        self.assertGreater(
            metrics["accuracy"], 0.23
        )  # 0.3333 for 60 questions; 0.234 for 1319 questions
        assert self.process.poll() is None

    def test_logprob_spec_v2(self):
        """Integration test: overlap spec v2 + EAGLE returns output_token_logprobs."""
        output_len = 8
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello, say one word:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        res = response.json()
        self.assertIn("meta_info", res)
        meta = res["meta_info"]
        self.assertIn("output_token_logprobs", meta)
        self.assertEqual(
            len(meta["output_token_logprobs"]),
            meta["completion_tokens"],
            "output_token_logprobs length must equal completion_tokens",
        )
        self.assertEqual(meta["completion_tokens"], output_len)
        # Each entry is [token_id, logprob]
        for i, entry in enumerate(meta["output_token_logprobs"]):
            self.assertIsInstance(entry, list, f"entry {i}")
            self.assertEqual(len(entry), 2, f"entry {i}: [token_id, logprob]")
            self.assertIsInstance(entry[0], int)
            self.assertIsInstance(entry[1], (int, float))
        assert self.process.poll() is None

    def test_logprob_spec_v2_top_logprobs(self):
        """Integration test: spec v2 + return_logprob + top_logprobs_num."""
        output_len = 4
        top_logprobs_num = 3
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Count: one two",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": top_logprobs_num,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        res = response.json()
        meta = res["meta_info"]
        self.assertEqual(len(meta["output_token_logprobs"]), output_len)
        self.assertIn("output_top_logprobs", meta)
        self.assertEqual(len(meta["output_top_logprobs"]), output_len)
        for i in range(output_len):
            self.assertEqual(
                len(meta["output_top_logprobs"][i]),
                top_logprobs_num,
                f"position {i}",
            )
        assert self.process.poll() is None


class TestEagleServerPage(TestEagleServerBase):
    other_launch_args = ["--page-size", "64"]


if __name__ == "__main__":
    unittest.main()
