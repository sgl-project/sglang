"""Small row pools expose mixed-width admission errors before memory pressure."""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")


class TestBeamSearchRegressions(CustomTestCase):
    def _generate(self, payload):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate", json=payload, timeout=120
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _check_mixed_width_admission(self):
        for widths in ([4, 1], [2, 1, 2], [1, 4]):
            with self.subTest(widths=widths):
                output = self._generate(
                    {
                        "text": ["Explain how rain forms."] * len(widths),
                        "sampling_params": [
                            {
                                "beam_width": width,
                                "n": 1,
                                "max_new_tokens": 8,
                                "temperature": 0,
                                "ignore_eos": True,
                            }
                            for width in widths
                        ],
                    }
                )
                self.assertEqual(len(output), len(widths))
                for result, width in zip(output, widths):
                    self.assertEqual(len(result["output_ids"]), 8)
                    if width > 1:
                        self.assertEqual(len(result["meta_info"]["beam_results"]), 1)
                    else:
                        self.assertNotIn("beam_results", result["meta_info"])

    def _check_stop_on_final_step(self):
        params = {
            "beam_width": 2,
            "n": 2,
            "max_new_tokens": 1,
            "ignore_eos": True,
            "skip_special_tokens": False,
        }
        request = {"text": "The capital of France is", "sampling_params": params}
        probe = self._generate(request)
        stop_id = probe["meta_info"]["beam_results"][0]["output_ids"][0]
        params.update(ignore_eos=False, stop_token_ids=[stop_id])
        output = self._generate(request)
        best = output["meta_info"]["beam_results"][0]
        self.assertEqual(best["output_ids"], [stop_id])
        self.assertEqual(best["meta_info"]["finish_reason"]["type"], "stop")
        self.assertEqual(best["text"], "")

    def test_small_pool_with_and_without_overlap(self):
        model = os.environ.get("SGLANG_TEST_BEAM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                args = [
                    "--page-size",
                    "1",
                    "--max-running-requests",
                    "4",
                    "--max-total-tokens",
                    "4096",
                    "--context-length",
                    "1024",
                    "--mem-fraction-static",
                    "0.6",
                    "--attention-backend",
                    "flashinfer",
                    "--sampling-backend",
                    "pytorch",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    # One IPC batch makes [width4, normal] hit the same gate.
                    "--enable-tokenizer-batch-encode",
                ]
                if not overlap:
                    args.append("--disable-overlap-schedule")
                process = popen_launch_server(
                    model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                )
                try:
                    self._check_mixed_width_admission()
                    self._check_stop_on_final_step()
                    recovered = self._generate(
                        {
                            "text": "Hello",
                            "sampling_params": {
                                "max_new_tokens": 2,
                                "ignore_eos": True,
                            },
                        }
                    )
                    self.assertEqual(len(recovered["output_ids"]), 2)
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
