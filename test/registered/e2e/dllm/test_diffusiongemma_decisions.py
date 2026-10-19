"""Decision scores must follow the empty thought prefix, with replayable prompts."""

import math
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="nightly", runner_config="1-gpu-large")


class TestDiffusionGemmaDecisions(CustomTestCase):
    model = "google/diffusiongemma-26B-A4B-it"
    base_url = DEFAULT_URL_FOR_TEST
    scheduling_args = []

    @classmethod
    def setUpClass(cls):
        cls.config_dir = tempfile.TemporaryDirectory()
        config = Path(cls.config_dir.name) / "renoise.yaml"
        # A fixed sampler seed makes raw-score replay independent of request ids.
        # Keep the checkpoint's full canvas and default denoising schedule.
        config.write_text("seed: 42\n")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--model-type",
                "llm",
                "--dllm-algorithm",
                "Gemma4Renoise",
                "--dllm-algorithm-config",
                str(config),
                "--trust-remote-code",
                "--context-length",
                "2048",
                "--max-running-requests",
                "1",
                "--mem-fraction-static",
                "0.85",
                "--cuda-graph-bs-decode",
                "1",
                *cls.scheduling_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "config_dir"):
            cls.config_dir.cleanup()

    def post(self, route, body):
        response = requests.post(self.base_url + route, json=body, timeout=300)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_decisions_replay_and_raw_canvas_alignment(self):
        temperature = 1.7
        body = {
            "input": "The customer's payment was declined. They need help today.",
            "questions": [
                {
                    "id": "team",
                    "type": "choice",
                    "question": "Which team should handle the payment problem?",
                    "options": [{"name": "billing"}, {"name": "technical"}],
                },
                {
                    "id": "urgency",
                    "type": "score",
                    "question": "How soon does the customer need help?",
                    "levels": ["Can wait a month", "Needs help today"],
                },
                {
                    "id": "today",
                    "type": "yes_no",
                    "question": "The customer needs help today.",
                },
            ],
            "temperature": temperature,
            "return_prompt_token_ids": True,
        }
        result = self.post("/v1/decisions", body)
        self.assertEqual(result["usage"]["completion_tokens"], 0)
        self.assertEqual(
            result["usage"]["prompt_tokens"],
            sum(len(a["prompt_token_ids"]) for a in result["answers"].values()),
        )
        for answer in result["answers"].values():
            with self.subTest(kind=answer["type"]):
                probabilities = list(answer["probabilities"].values())
                self.assertAlmostEqual(sum(probabilities), 1.0)
                self.assertTrue(
                    all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                )
                self.assertGreater(answer["label_mass"], 0)
                self.assertLessEqual(answer["label_mass"], 1.00001)
                replay = self.post(
                    "/v1/score",
                    {
                        "query": [],
                        "items": [answer["prompt_token_ids"]],
                        "label_token_ids": [answer["label_token_ids"]],
                        "apply_softmax": True,
                        "temperature": temperature,
                        "return_token_logprobs": True,
                    },
                )
                raw = self.post(
                    "/generate",
                    {
                        "input_ids": answer["prompt_token_ids"],
                        "sampling_params": {"max_new_tokens": 5, "ignore_eos": True},
                        "return_logprob": True,
                        "token_ids_logprob": answer["label_token_ids"],
                    },
                )
                self.assertEqual(raw["output_ids"][:4], [100, 45518, 107, 101])
                logprobs = [
                    lp for lp, _, _ in raw["meta_info"]["output_token_ids_logprobs"][4]
                ]
                maximum = max(logprobs)
                weights = [math.exp((lp - maximum) / temperature) for lp in logprobs]
                expected = [w / sum(weights) for w in weights]
                for actual, replayed, reference in zip(
                    probabilities, replay["scores"][0], expected
                ):
                    self.assertAlmostEqual(actual, replayed, delta=2e-4)
                    self.assertAlmostEqual(actual, reference, delta=2e-4)
                self.assertAlmostEqual(
                    answer["label_mass"],
                    math.fsum(math.exp(lp) for lp in logprobs),
                    delta=2e-4,
                )
        self.assertEqual(result["answers"]["team"]["choice"], "billing")
        self.assertGreater(result["answers"]["today"]["probabilities"]["yes"], 0.5)
        self.assertGreater(result["answers"]["urgency"]["score"], 0.5)

    def test_systemone_shares_diffusion_scoring(self):
        result = self.post(
            "/v1/systemone",
            {
                "model": self.model,
                "state": "The customer needs help today.",
                "questions": {
                    "today": {"type": "noul", "instructions": "Help is needed today."}
                },
            },
        )
        self.assertGreater(result["answers"]["today"]["noul"], 0.5)
        self.assertEqual(result["usage"]["output_tokens"], 0)


class TestDiffusionGemmaDecisionsSync(TestDiffusionGemmaDecisions):
    scheduling_args = ["--no-dllm-fdfo", "--cuda-graph-backend-decode", "disabled"]


if __name__ == "__main__":
    unittest.main()
