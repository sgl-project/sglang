import unittest
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=369, stage="stage-b", runner_config="1-gpu-small")


class TestEagle3ServerBase(CustomTestCase, MatchedStopMixin):
    max_running_requests = 64
    attention_backend = "triton"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 1
    other_launch_args = []
    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--dtype=float16",
            "--chunked-prefill-size",
            "1024",
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
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        with (
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_SPEC_NAN_DETECTION.override(True),
            envs.SGLANG_SPEC_OOB_DETECTION.override(True),
            envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True),
        ):
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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1000,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"TestEagle3LargeBS -- {metrics=}")
        self.assertGreater(metrics["score"], 0.7)
        assert self.process.poll() is None

    def test_logprob_spec_v2_match(self):
        """Verify spec v2 decode logprobs match prefill scoring logprobs.

        Generate tokens with spec v2, then score the same sequence via
        prefill-only (no speculation). The two sets of logprobs should be
        close, validating that spec v2 computes logprobs correctly.

        Runs two rounds with different prompts to catch state-dependent bugs.
        """
        top_k = 5
        probe_token_ids = [1, 2, 10, 100, 1000]
        prompts = [
            "The capital of France is",
            "Explain quantum computing in simple terms:",
        ]

        for round_idx, prompt in enumerate(prompts):
            with self.subTest(round=round_idx, prompt=prompt):
                gen_res = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                ).json()

                decode_logprobs = gen_res["meta_info"]["output_token_logprobs"]
                decode_top_logprobs = gen_res["meta_info"]["output_top_logprobs"]
                decode_tid_logprobs = gen_res["meta_info"]["output_token_ids_logprobs"]
                input_token_ids = [
                    t[1] for t in gen_res["meta_info"]["input_token_logprobs"]
                ]
                output_token_ids = [t[1] for t in decode_logprobs]
                num_prompt_tokens = gen_res["meta_info"]["prompt_tokens"]

                score_res = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": input_token_ids + output_token_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 0,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                ).json()

                score_logprobs = score_res["meta_info"]["input_token_logprobs"][
                    num_prompt_tokens:
                ]
                score_top_logprobs = score_res["meta_info"]["input_top_logprobs"][
                    num_prompt_tokens:
                ]
                score_tid_logprobs = score_res["meta_info"]["input_token_ids_logprobs"][
                    num_prompt_tokens:
                ]

                self.assertEqual(len(decode_logprobs), len(score_logprobs))

                # Check per-token logprobs
                decode_vals = np.array([t[0] for t in decode_logprobs])
                score_vals = np.array([t[0] for t in score_logprobs])
                max_diff = np.max(np.abs(decode_vals - score_vals))
                print(
                    f"[round {round_idx}] prompt={prompt!r} "
                    f"logprob max_diff={max_diff:.6f}"
                )
                print(f"[round {round_idx}] decode_vals[-5:]={decode_vals[-5:]}")
                print(f"[round {round_idx}] score_vals[-5:]={score_vals[-5:]}")
                self.assertLess(max_diff, 0.255)

                # Check top-k logprobs
                for pos in range(len(decode_logprobs)):
                    dec_top = {t[1]: t[0] for t in decode_top_logprobs[pos]}
                    scr_top = {t[1]: t[0] for t in score_top_logprobs[pos]}
                    common_ids = set(dec_top.keys()) & set(scr_top.keys())
                    self.assertGreater(len(common_ids), 0)
                    for tid in common_ids:
                        self.assertAlmostEqual(dec_top[tid], scr_top[tid], delta=0.255)

                # Check token_ids_logprob
                self.assertEqual(len(decode_tid_logprobs), len(score_tid_logprobs))
                for pos in range(len(decode_tid_logprobs)):
                    dec_tid = {t[1]: t[0] for t in decode_tid_logprobs[pos]}
                    scr_tid = {t[1]: t[0] for t in score_tid_logprobs[pos]}
                    self.assertEqual(set(dec_tid.keys()), set(scr_tid.keys()))
                    for tid in dec_tid:
                        self.assertAlmostEqual(dec_tid[tid], scr_tid[tid], delta=0.255)

    def test_token_ids_logprob_ragged(self):
        """Regression: get_token_ids_logprobs_raw crashes on ragged token_ids_logprob lists.

        Sends concurrent requests with different-length token_ids_logprob lists
        so they land in the same batch. torch.tensor() on ragged input will crash.
        """
        import concurrent.futures

        def send(probe_ids):
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Hello world",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                    "return_logprob": True,
                    "top_logprobs_num": 3,
                    "token_ids_logprob": probe_ids,
                },
            ).json()

        ragged_probes = [
            [1, 2],
            [3, 4, 5],
            [6],
            [10, 20, 30, 40],
            [1, 2],
            [3, 4, 5],
            [6],
            [10, 20, 30, 40],
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(send, ids) for ids in ragged_probes]
            for f in concurrent.futures.as_completed(futs):
                res = f.result()
                self.assertIn("text", res, f"Server error: {res}")

    def test_penalty(self):
        """Verify spec v2 handles penalty parameters without crashing."""
        import concurrent.futures

        args = [
            {"max_new_tokens": 32},
            {"max_new_tokens": 16, "frequency_penalty": 2},
            {"max_new_tokens": 48, "presence_penalty": 1},
            {"max_new_tokens": 8, "frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"max_new_tokens": 64, "frequency_penalty": -0.5, "presence_penalty": 0.3},
            {"max_new_tokens": 24, "min_new_tokens": 8, "frequency_penalty": 0.4},
            {"max_new_tokens": 32, "repetition_penalty": 1.5},
        ]

        def run_decode(sampling_params):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": sampling_params,
                },
            )
            self.assertEqual(response.status_code, 200)
            res = response.json()
            self.assertIn("text", res, f"Server error: {res}")
            self.assertIsInstance(
                res["text"],
                str,
                f"Expected 'text' to be str, got {type(res['text']).__name__}: {res}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run_decode, args * 3))
        assert self.process.poll() is None


class TestEagle3ServerPage(TestEagle3ServerBase):
    other_launch_args = ["--page-size", "64"]


if __name__ == "__main__":
    unittest.main()
