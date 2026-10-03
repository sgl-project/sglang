"""Pinned Clef GPU regression against small synthetic native-reference answers."""

import copy
import json
import math
import os
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.srt.layers.clef_reference import encode_record
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


def decisions_request(native):
    """Translate the fixture independently of the server's request adapter."""
    questions = []
    for qid, question in native["questions"].items():
        item = {"id": qid, "question": question["instructions"]}
        if question["type"] == "noul":
            item["type"] = "yes_no"
        elif question["type"] == "choice":
            item.update(
                type="choice",
                options=[
                    {"name": name, "description": description}
                    for name, description in question["criteria"].items()
                ],
            )
        else:
            item.update(type="score", levels=question["criteria"])
        questions.append(item)
    return {
        "model": native["model"],
        "input": native["state"],
        "questions": questions,
        "temperature": 1,
        "prompt_format_version": 1,
        "return_prompt_token_ids": True,
    }


class TestClefDecisions(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(
            Path(__file__)
            .with_name("fixtures")
            .joinpath("clef_native.json")
            .read_text()
        )
        # Optional local copy must contain the same pinned checkpoint files.
        model = os.environ.get("CLEF_TEST_MODEL_PATH", cls.fixture["model"])
        cls.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=cls.fixture["revision"]
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_RUST_SERVER": "0"},
            other_args=[
                "--revision",
                cls.fixture["revision"],
                "--served-model-name",
                "clef-flash",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "triton",
                "--linear-attn-backend",
                "triton",
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.82",
                "--max-running-requests",
                "1",
                "--chunked-prefill-size",
                "-1",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
            ],
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def post(self, route, body):
        response = requests.post(self.base_url + route, json=body, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["execution"]["path"], "sglang_backbone_joint_head")
        self.assertGreater(result["execution"]["forward_count"], 0)
        return result

    def test_native_reference_on_both_routes(self):
        previous_count = 0
        for case in self.fixture["cases"]:
            native = case["request"]
            encoded = encode_record(self.tokenizer, native, max_length=2**63 - 1)
            self.assertEqual(len(encoded.input_ids), case["input_tokens"])
            for route, payload in (
                ("/v1/decisions", decisions_request(native)),
                ("/v1/systemone", native),
            ):
                with self.subTest(route=route, state=native["state"]):
                    result = self.post(route, payload)
                    self.assertGreater(
                        result["execution"]["forward_count"], previous_count
                    )
                    previous_count = result["execution"]["forward_count"]
                    self.assertEqual(set(result["answers"]), set(native["questions"]))
                    decisions = route == "/v1/decisions"
                    self.assertEqual(
                        result["usage"][
                            "prompt_tokens" if decisions else "input_tokens"
                        ],
                        case["input_tokens"],
                    )
                    self.assertEqual(
                        result["usage"][
                            "completion_tokens" if decisions else "output_tokens"
                        ],
                        0,
                    )
                    for qid, golden in case["answers"].items():
                        actual = result["answers"][qid]
                        self.assertNotIn("label_mass", actual)
                        self.assertNotIn("x_label_mass", actual)
                        self.assertNotIn("label_token_ids", actual)
                        if decisions:
                            self.assertEqual(
                                actual["prompt_token_ids"], list(encoded.input_ids)
                            )
                        if golden["type"] == "noul":
                            probability = (
                                actual["probabilities"]["yes"]
                                if decisions
                                else actual["noul"]
                            )
                            self.assertAlmostEqual(
                                probability, golden["noul"], delta=0.02
                            )
                        else:
                            probs = actual["probabilities"]
                            self.assertEqual(set(probs), set(golden["probabilities"]))
                            self.assertAlmostEqual(
                                math.fsum(probs.values()),
                                1,
                                delta=1e-8 if decisions else 0.0002,
                            )
                            for key, expected in golden["probabilities"].items():
                                self.assertAlmostEqual(probs[key], expected, delta=0.02)
                            if golden["type"] == "choice":
                                self.assertEqual(actual["choice"], golden["choice"])
                            else:
                                self.assertAlmostEqual(
                                    actual["score"], golden["score"], delta=0.04
                                )

    def test_empty_state_and_77_options(self):
        native = {
            "model": "clef-flash",
            "state": {},
            "questions": {
                "number": {
                    "type": "choice",
                    "instructions": "Choose the number seventy-six.",
                    "criteria": {f"option_{i}": str(i) for i in range(77)},
                }
            },
        }
        for route, body in (
            ("/v1/decisions", decisions_request(native)),
            ("/v1/systemone", native),
        ):
            with self.subTest(route=route):
                result = self.post(route, body)
                answer = result["answers"]["number"]
                self.assertEqual(
                    set(answer["probabilities"]),
                    set(native["questions"]["number"]["criteria"]),
                )
                self.assertEqual(answer["choice"], "option_76")

    def test_unsupported_and_overlength_requests(self):
        valid = decisions_request(self.fixture["cases"][0]["request"])
        before = self.post("/v1/decisions", valid)["execution"]["forward_count"]
        for field, value in (
            ("temperature", 0.5),
            ("images", ["unused.png"]),
            ("input", "overflow " * 5000),
        ):
            with self.subTest(field=field):
                body = copy.deepcopy(valid)
                body[field] = value
                response = requests.post(
                    self.base_url + "/v1/decisions", json=body, timeout=120
                )
                self.assertEqual(response.status_code, 400, response.text)
        after = self.post("/v1/decisions", valid)["execution"]["forward_count"]
        self.assertEqual(after, before + 1)


if __name__ == "__main__":
    unittest.main()
