import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class TestNemotronLabsDiffusion(CustomTestCase):
    def test_ar_and_diffusion_serve_distinct_concurrent_requests(self):
        model = "nvidia/Nemotron-Labs-Diffusion-8B"
        url = DEFAULT_URL_FOR_TEST
        cases = [
            ("Reply with just the answer. What is 15% of 240?", "36"),
            ("Reply with just the name. What is the capital of France?", "Paris"),
            ("Translate to French: The cat is sleeping.", "chat"),
            ("Explain in one short sentence why ice floats on water.", "less dense"),
        ]
        for algorithm in (None, "NemotronFastDiffuser"):
            args = [
                "--context-length",
                "4096",
                "--max-running-requests",
                "4",
                "--max-total-tokens",
                "8192",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
            ]
            if algorithm is not None:
                args += ["--dllm-algorithm", algorithm]
            process = popen_launch_server(
                model, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=args
            )
            try:

                def generate(case):
                    question, answer = case
                    response = requests.post(
                        url + "/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": question}],
                            "temperature": 0,
                            "max_tokens": 256,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    text = response.json()["choices"][0]["message"]["content"]
                    self.assertIn(answer, text)
                    return text

                with self.subTest(algorithm=algorithm):
                    for case in cases:
                        generate(case)
                    for concurrency in (2, 3, 4):
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            list(pool.map(generate, cases[:concurrency]))
                    for limit in (1, 31, 33):
                        response = requests.post(
                            url + "/generate",
                            json={
                                "text": "Count from one to one hundred:",
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": limit,
                                    "ignore_eos": True,
                                },
                            },
                            timeout=180,
                        )
                        response.raise_for_status()
                        data = response.json()
                        self.assertEqual(data["meta_info"]["completion_tokens"], limit)
                        self.assertNotIn(100, data["output_ids"])
                    if algorithm is not None:
                        for unsupported in (
                            {"temperature": 1},
                            {"frequency_penalty": 0.5},
                            {"logprobs": True},
                        ):
                            with self.subTest(unsupported=unsupported):
                                response = requests.post(
                                    url + "/v1/chat/completions",
                                    json={
                                        "model": model,
                                        "messages": [
                                            {"role": "user", "content": cases[0][0]}
                                        ],
                                        "temperature": 0,
                                        "max_tokens": 32,
                                        **unsupported,
                                    },
                                    timeout=180,
                                )
                                self.assertEqual(response.status_code, 400)
                                self.assertIn(
                                    "NemotronFastDiffuser supports greedy decoding",
                                    response.text,
                                )
                        generate(cases[0])
            finally:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
