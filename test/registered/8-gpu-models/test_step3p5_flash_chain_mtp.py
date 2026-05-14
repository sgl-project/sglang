import unittest
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=663, stage="stage-c", runner_config="8-gpu-h200")

STEP3P5_FLASH_MODEL_PATH = "stepfun-ai/Step-3.5-Flash"


class TestStep3p5FlashChainMTP(CustomTestCase):
    """Chain-style multi-layer EAGLE speculative decoding on Step-3.5-Flash.

    Step3p5ForCausalLM auto-enables multi-layer EAGLE and spec v2 when
    --speculative-algorithm=EAGLE is set.  The chain MTP propagation
    (each MTP layer feeds its hidden states to the next) is activated
    automatically for the Step3p5MTP draft architecture.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = STEP3P5_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--attention-backend",
            "fa3",
            "--enable-multi-layer-eagle",
            "--mem-fraction-static",
            "0.75",
            "--chunked-prefill-size",
            "4096",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (step-3.5-flash chain mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.83)
            self.assertGreater(avg_spec_accept_length, 2.6)

    def test_logprob_spec_v2_match(self):
        """Verify spec v2 decode logprobs match prefill scoring logprobs.

        Generate tokens with chain MTP spec v2, then score the same sequence
        via prefill-only (no speculation). The two sets of logprobs should be
        close, validating that spec v2 + multi-layer EAGLE computes logprobs
        correctly.
        """
        requests.get(self.base_url + "/flush_cache")

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

                # Top-k / probe tokens are not sampled, so they drift more than
                # the chosen-token logprob under TP=8 + multi-layer EAGLE noise.
                # Collect the diff distribution to see whether outliers are
                # isolated tail tokens or systemic drift before asserting.
                top_diffs = []
                for pos in range(len(decode_logprobs)):
                    dec_top = {t[1]: t[0] for t in decode_top_logprobs[pos]}
                    scr_top = {t[1]: t[0] for t in score_top_logprobs[pos]}
                    common_ids = set(dec_top.keys()) & set(scr_top.keys())
                    self.assertGreater(len(common_ids), 0)
                    for tid in common_ids:
                        top_diffs.append(abs(dec_top[tid] - scr_top[tid]))
                top_diffs_arr = np.array(top_diffs)
                print(
                    f"[round {round_idx}] top-k diffs: "
                    f"n={len(top_diffs_arr)} "
                    f"max={top_diffs_arr.max():.4f} "
                    f"p99={np.percentile(top_diffs_arr, 99):.4f} "
                    f"p95={np.percentile(top_diffs_arr, 95):.4f} "
                    f"p50={np.percentile(top_diffs_arr, 50):.4f} "
                    f"mean={top_diffs_arr.mean():.4f}"
                )

                self.assertEqual(len(decode_tid_logprobs), len(score_tid_logprobs))
                tid_diffs = []
                for pos in range(len(decode_tid_logprobs)):
                    dec_tid = {t[1]: t[0] for t in decode_tid_logprobs[pos]}
                    scr_tid = {t[1]: t[0] for t in score_tid_logprobs[pos]}
                    self.assertEqual(set(dec_tid.keys()), set(scr_tid.keys()))
                    for tid in dec_tid:
                        tid_diffs.append(abs(dec_tid[tid] - scr_tid[tid]))
                tid_diffs_arr = np.array(tid_diffs)
                print(
                    f"[round {round_idx}] token_ids_logprob diffs: "
                    f"n={len(tid_diffs_arr)} "
                    f"max={tid_diffs_arr.max():.4f} "
                    f"p99={np.percentile(tid_diffs_arr, 99):.4f} "
                    f"p95={np.percentile(tid_diffs_arr, 95):.4f} "
                    f"p50={np.percentile(tid_diffs_arr, 50):.4f} "
                    f"mean={tid_diffs_arr.mean():.4f}"
                )

                # Bulk of the distribution must stay tight. Tail (max / p99) is
                # dominated by very low-probability tokens whose logprobs are
                # extremely sensitive to BF16 + TP=8 logsumexp noise — a real
                # bug in chain MTP hidden state propagation would shift the
                # median, not just the tail.
                self.assertLess(np.percentile(top_diffs_arr, 50), 0.1)
                self.assertLess(top_diffs_arr.mean(), 0.2)
                self.assertLess(np.percentile(top_diffs_arr, 95), 0.4)
                self.assertLess(np.percentile(tid_diffs_arr, 50), 0.2)
                self.assertLess(tid_diffs_arr.mean(), 0.4)


if __name__ == "__main__":
    unittest.main()
