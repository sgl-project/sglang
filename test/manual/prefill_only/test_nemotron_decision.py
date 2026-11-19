import math
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Decision-8B-BF16"
REVISION = "40c36c022455a731f1226594108dac296098102f"
CASES = (
    ("france", {"capital": "Paris"}, "Which capital is listed?", "Paris", "Berlin"),
    (
        "germany",
        {"Stadt": "Berlin"},
        "Welche Stadt ist angegeben?",  # codespell:ignore ist
        "Berlin",
        "Paris",
    ),
    ("canada", {"ville": "Montréal"}, "Quelle ville est indiquée?", "Montréal", "Lyon"),
    (
        "ticket",
        {"team": "billing", "message": "Charged twice"},
        "Which team is named?",
        "billing",
        "sales",
    ),
)


class TestNemotronDecisionServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.url = os.environ.get("NEMOTRON_DECISION_TEST_URL", DEFAULT_URL_FOR_TEST)
        cls.process = None
        if "NEMOTRON_DECISION_TEST_URL" not in os.environ:
            cls.process = popen_launch_server(
                os.environ.get("NEMOTRON_DECISION_MODEL_PATH", MODEL),
                cls.url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--revision",
                    REVISION,
                    "--served-model-name",
                    "decision-test",
                    "--context-length",
                    "16384",
                    "--max-prefill-tokens",
                    "16384",
                    "--max-total-tokens",
                    "16384",
                    "--max-running-requests",
                    "4",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        super().tearDownClass()

    def request_decision(self, case, route="/v1/systemone"):
        key, state, question, expected, other = case
        payload = {
            "model": "decision-test",
            "state": state,
            "questions": {
                key: {
                    "type": "choice",
                    "instructions": question,
                    "criteria": {
                        expected: expected,
                        other: other,
                        "unknown": "Not given",
                    },
                },
                key + "-present": {
                    "type": "noul",
                    "instructions": f"Does the state mention {expected}?",
                },
            },
        }
        if route == "/v1/decisions":
            payload = {
                "model": "decision-test",
                "input": state,
                "questions": [
                    {
                        "id": key,
                        "type": "choice",
                        "question": question,
                        "options": [
                            {"name": name, "description": meaning}
                            for name, meaning in payload["questions"][key][
                                "criteria"
                            ].items()
                        ],
                    },
                    {
                        "id": key + "-present",
                        "type": "yes_no",
                        "question": f"Does the state mention {expected}?",
                    },
                ],
            }
        response = requests.post(self.url + route, json=payload, timeout=120)
        response.raise_for_status()
        answers = response.json()["answers"]
        self.assertEqual(set(answers), {key, key + "-present"})
        answer = answers[key]
        self.assertEqual(answer["choice"], expected)
        self.assertEqual(set(answer["probabilities"]), {expected, other, "unknown"})
        self.assertTrue(
            all(
                math.isfinite(p) and 0 <= p <= 1
                for p in answer["probabilities"].values()
            )
        )
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
        present = answers[key + "-present"]
        probability = (
            present["noul"]
            if route == "/v1/systemone"
            else present["probabilities"]["yes"]
        )
        self.assertGreater(probability, 0.5)
        return answer["probabilities"]

    def test_distinct_questions_keep_their_answers_under_concurrency(self):
        expected = [self.request_decision(case) for case in CASES]
        for route in ("/v1/systemone", "/v1/decisions"):
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual = list(
                    pool.map(lambda case: self.request_decision(case, route), CASES)
                )
            for serial, concurrent in zip(expected, actual):
                for label in serial:
                    self.assertAlmostEqual(serial[label], concurrent[label], delta=0.02)

    def test_probabilities_match_bidirectional_reference(self):
        criteria = {f"item{i:03d}": f"Item {i:03d}" for i in range(501)}
        response = requests.post(
            self.url + "/v1/systemone",
            json={
                "model": "decision-test",
                "state": {"selected": "item499"},
                "questions": {
                    "selected": {
                        "type": "choice",
                        "instructions": "Which item is selected?",
                        "criteria": criteria,
                    }
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        actual = response.json()["answers"]["selected"]["probabilities"]
        self.assertEqual(set(actual), set(criteria))
        self.assertEqual(max(actual, key=actual.get), "item499")
        # The reference is 0.7478; causal attention gives 0.6921.
        self.assertAlmostEqual(actual["item499"], 0.7477865, delta=0.02)

    def test_score_order_and_prompt_version_validation(self):
        response = requests.post(
            self.url + "/v1/systemone",
            json={
                "model": "decision-test",
                "state": "The service was excellent. Everything worked perfectly.",
                "questions": {
                    "sentiment": {
                        "type": "score",
                        "instructions": "Rate the sentiment.",
                        "criteria": ["negative", "neutral", "positive"],
                    }
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["answers"]["sentiment"]
        self.assertEqual(
            max(answer["probabilities"], key=answer["probabilities"].get), "2"
        )
        payload = {
            "model": "decision-test",
            "input": "The service was excellent. Everything worked perfectly.",
            "return_prompt_token_ids": True,
            "questions": [
                {
                    "id": "sentiment",
                    "type": "score",
                    "question": "Rate the sentiment.",
                    "levels": ["negative", "neutral", "positive"],
                }
            ],
        }
        response = requests.post(self.url + "/v1/decisions", json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        self.assertEqual(body["prompt_format_version"], 2)
        actual = body["answers"]["sentiment"]
        for label, expected in answer["probabilities"].items():
            self.assertAlmostEqual(actual["probabilities"][label], expected, delta=0.02)
        self.assertEqual(len(actual["label_token_ids"]), 3)
        self.assertTrue(actual["prompt_token_ids"])
        payload["prompt_format_version"] = 1
        response = requests.post(self.url + "/v1/decisions", json=payload, timeout=30)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("uses version 2", response.text)

    def test_unsupported_modes_and_unschedulable_prompts_do_not_block_valid_reads(self):
        payload = {
            "model": "decision-test",
            "state": "Paris",
            "questions": {
                "city": {
                    "type": "choice",
                    "instructions": "Which city?",
                    "criteria": {"Paris": "Paris", "Lyon": "Lyon"},
                }
            },
        }
        for extension in ({"packing": "shared"}, {"long_mode": "compact"}):
            response = requests.post(
                self.url + "/v1/systemone", json={**payload, **extension}, timeout=30
            )
            self.assertEqual(response.status_code, 400, response.text)
        info = requests.get(self.url + "/server_info", timeout=30)
        info.raise_for_status()
        oversized = info.json()["max_prefill_tokens"] + 1
        for length, max_tokens, error in (
            (oversized, 0, "complete prompt"),
            (16, 2, "single-token"),
        ):
            response = requests.post(
                self.url + "/generate",
                json={
                    "input_ids": [1] * length,
                    "sampling_params": {"max_new_tokens": max_tokens},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn(error, response.text)
        self.request_decision(CASES[0])


if __name__ == "__main__":
    unittest.main()
