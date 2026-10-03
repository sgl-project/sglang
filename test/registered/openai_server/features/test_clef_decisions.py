"""Pinned Clef GPU regression against small synthetic native-reference answers."""

import copy
import json
import math
import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        directory = tempfile.TemporaryDirectory(prefix="clef-serving-")
        cls.addClassCleanup(directory.cleanup)
        cls.log_path = Path(directory.name) / "server.log"
        log = cls.log_path.open("w", buffering=1)
        cls.addClassCleanup(log.close)
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_RUST_SERVER": "0"},
            return_stdout_stderr=(log, log),
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
                "0.86",
                "--max-running-requests",
                "4",
                "--max-mamba-cache-size",
                "24",
                "--chunked-prefill-size",
                "256",
                "--cuda-graph-max-bs-decode",
                "4",
                "--disable-prefill-cuda-graph",
            ],
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def post(self, route, body):
        response = requests.post(self.base_url + route, json=body, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        return result

    def test_native_reference_on_both_routes(self):
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
                    self.assert_native(case, result, route)

    def assert_native(self, case, result, route):
        native = case["request"]
        encoded = encode_record(self.tokenizer, native, max_length=2**63 - 1)
        self.assertEqual(set(result["answers"]), set(native["questions"]))
        decisions = route == "/v1/decisions"
        self.assertEqual(
            result["usage"]["prompt_tokens" if decisions else "input_tokens"],
            case["input_tokens"],
        )
        self.assertEqual(
            result["usage"]["completion_tokens" if decisions else "output_tokens"],
            0,
        )
        for qid, golden in case["answers"].items():
            actual = result["answers"][qid]
            self.assertNotIn("label_mass", actual)
            self.assertNotIn("x_label_mass", actual)
            self.assertNotIn("label_token_ids", actual)
            if decisions:
                self.assertEqual(actual["prompt_token_ids"], list(encoded.input_ids))
            if golden["type"] == "noul":
                probability = (
                    actual["probabilities"]["yes"] if decisions else actual["noul"]
                )
                self.assertAlmostEqual(probability, golden["noul"], delta=0.02)
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
                    self.assertAlmostEqual(actual["score"], golden["score"], delta=0.04)

    def flush_cache(self):
        response = requests.post(
            self.base_url + "/flush_cache", params={"timeout": 10}, timeout=20
        )
        self.assertEqual(response.status_code, 200, response.text)

    def prefill_events(self, offset):
        text = self.log_path.read_text()[offset:]
        return [
            tuple(map(int, match))
            for match in re.findall(
                r"Prefill batch(?: \[\d+\])?, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)",
                text,
            )
        ]

    def wait_prefill_events(self, offset, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            events = self.prefill_events(offset)
            if condition(events):
                return events
            time.sleep(0.02)
        self.fail(
            f"Required prefill execution not observed: {self.prefill_events(offset)}"
        )

    def test_chunked_and_cached_native_request(self):
        case = self.fixture["cases"][0]
        payload = decisions_request(case["request"])
        self.flush_cache()
        offset = len(self.log_path.read_text())
        cold = self.post("/v1/decisions", payload)
        self.assert_native(case, cold, "/v1/decisions")
        cold_events = self.wait_prefill_events(
            offset, lambda events: sum(e[1] for e in events) >= case["input_tokens"]
        )
        self.assertGreater(len(cold_events), 1)
        self.assertEqual(sum(e[1] for e in cold_events), case["input_tokens"])
        offset = len(self.log_path.read_text())
        warm = self.post("/v1/decisions", payload)
        self.assert_native(case, warm, "/v1/decisions")
        warm_events = self.wait_prefill_events(offset, bool)
        self.assertGreater(sum(e[2] for e in warm_events), 0)
        self.assertLess(sum(e[1] for e in warm_events), case["input_tokens"])

    def test_concurrent_native_requests(self):
        cases = self.fixture["cases"]
        for attempt in range(3):
            self.flush_cache()
            offset = len(self.log_path.read_text())
            barrier = threading.Barrier(len(cases))

            def submit(index, case):
                route = "/v1/decisions" if index % 2 == 0 else "/v1/systemone"
                body = (
                    decisions_request(case["request"])
                    if index % 2 == 0
                    else case["request"]
                )
                barrier.wait(timeout=10)
                return route, self.post(route, body)

            with ThreadPoolExecutor(max_workers=len(cases)) as executor:
                futures = [
                    executor.submit(submit, i, case) for i, case in enumerate(cases)
                ]
                results = [future.result(timeout=120) for future in futures]
            for case, (route, result) in zip(cases, results):
                self.assert_native(case, result, route)
            events = self.wait_prefill_events(offset, bool)
            if any(event[0] >= 2 for event in events):
                return
        self.fail(f"Correct concurrent answers never shared a backbone batch: {events}")

    def test_abort_returns_error_and_releases_request(self):
        self.flush_cache()
        payload = decisions_request(self.fixture["cases"][0]["request"])
        payload["input"] = {"padding": "neutral invoice context " * 700}
        tokens = len(
            encode_record(
                self.tokenizer,
                {
                    "state": payload["input"],
                    "questions": self.fixture["cases"][0]["request"]["questions"],
                },
                max_length=2**63 - 1,
            ).input_ids
        )
        self.assertGreater(tokens, 256)
        self.assertLess(tokens, 4095)
        offset = len(self.log_path.read_text())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                requests.post,
                self.base_url + "/v1/decisions",
                json=payload,
                timeout=120,
            )
            self.wait_prefill_events(offset, bool)
            self.assertFalse(
                future.done(), "Request completed before cancellation precondition"
            )
            dispatched = requests.post(
                self.base_url + "/abort_request", json={"abort_all": True}, timeout=20
            )
            self.assertEqual(dispatched.status_code, 200, dispatched.text)
            response = future.result(timeout=120)
        self.assertEqual(response.status_code, 503, response.text)
        error = response.json()
        self.assertEqual(error["type"], "RequestAborted")
        self.assertEqual(error["code"], 503)
        self.assertNotIn("answers", error)
        self.flush_cache()
        case = self.fixture["cases"][0]
        self.assert_native(
            case,
            self.post("/v1/decisions", decisions_request(case["request"])),
            "/v1/decisions",
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
        for field, value, reason in (
            ("temperature", 0.5, "temperature=1"),
            ("images", ["unused.png"], "text only"),
            ("input", "overflow " * 5000, "exceeds context length"),
        ):
            with self.subTest(field=field):
                body = copy.deepcopy(valid)
                body[field] = value
                response = requests.post(
                    self.base_url + "/v1/decisions", json=body, timeout=120
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(reason, response.json()["message"])


if __name__ == "__main__":
    unittest.main()
