import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=180, stage="nightly", runner_config="2-gpu-large", nightly=True
)

_REQUEST_TIMEOUT = 180


def _assert_spans_contiguous(test, meta_info):
    spans = meta_info["weight_versions"]
    test.assertGreater(len(spans), 0)
    test.assertEqual(spans[0]["start"], 0)
    for prev, cur in zip(spans, spans[1:]):
        test.assertEqual(prev["end"], cur["start"])
        test.assertNotEqual(prev["version"], cur["version"])
    test.assertEqual(meta_info["weight_version"], spans[-1]["version"])
    return spans


class TestWeightVersionSpans(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--weight-version",
                "base-v0",
                "--trust-remote-code",
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_MLA,
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "3",
                "--speculative-num-draft-tokens",
                "3",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, max_new_tokens: int, prompt: str = "The capital of France is"):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _current_version(self):
        response = requests.get(f"{self.base_url}/get_model_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()["weight_version"]

    def _pause(self, mode: str):
        requests.post(
            f"{self.base_url}/pause_generation",
            json={"mode": mode},
            timeout=30,
        ).raise_for_status()

    def _continue(self):
        requests.post(
            f"{self.base_url}/continue_generation",
            json={},
            timeout=30,
        ).raise_for_status()

    def _set_weight_version(self, new_version: str, abort_all_requests: bool = False):
        response = requests.post(
            f"{self.base_url}/update_weight_version",
            json={
                "new_version": new_version,
                "abort_all_requests": abort_all_requests,
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _update_weights_from_disk(self, **fields) -> None:
        response = requests.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": self.model, "flush_cache": False, **fields},
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    def _run_while_paused(
        self,
        num_requests: int,
        while_paused,
        mode: str = "retract",
        max_new_tokens: int = 1024,
    ):
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [
                executor.submit(
                    self._generate,
                    max_new_tokens=max_new_tokens - 16 * i,
                    prompt=f"Write a long story about the number {i}.",
                )
                for i in range(num_requests)
            ]

            time.sleep(2)
            self._pause(mode)
            try:
                while_paused()
            finally:
                self._continue()

            return [future.result() for future in futures]

    def test_01_single_span_without_update(self):
        """A request untouched by updates reports one span covering all output tokens."""
        data = self._generate(max_new_tokens=8)

        meta_info = data["meta_info"]
        spans = _assert_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], "base-v0")
        self.assertEqual(spans[0]["end"], meta_info["completion_tokens"])
        self.assertEqual(meta_info["weight_version"], "base-v0")

    def test_02_update_weight_version_endpoint_applies_to_new_requests(self):
        """The endpoint returns only once every scheduler stamps new requests with the new version."""
        self._set_weight_version("endpoint-v1")
        self.assertEqual(self._current_version(), "endpoint-v1")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._generate, max_new_tokens=8) for _ in range(4)
            ]
            results = [future.result() for future in futures]

        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["version"], "endpoint-v1")

    def test_03_spans_split_across_pause_update_continue(self):
        """Requests spanning pause -> update_weights_from_disk -> continue report one span per version."""
        base_version = self._current_version()

        results = self._run_while_paused(
            num_requests=4,
            while_paused=lambda: self._update_weights_from_disk(
                weight_version="disk-v2"
            ),
        )

        multi_span_count = 0
        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            versions = [span["version"] for span in spans]
            self.assertEqual(versions[0], base_version)
            self.assertIn(versions[-1], (base_version, "disk-v2"))
            if len(spans) > 1:
                multi_span_count += 1
                self.assertEqual(versions, [base_version, "disk-v2"])
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            multi_span_count,
            0,
            "No request spanned the weight update -- no boundary was recorded.",
        )

    def test_04_openai_metadata_contains_weight_versions(self):
        """OpenAI-compatible responses surface the spans under response metadata."""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        metadata = data["metadata"]
        self.assertIn("weight_versions", metadata)
        spans = metadata["weight_versions"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], metadata["weight_version"])
        self.assertEqual(metadata["weight_version"], self._current_version())
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[0]["end"], data["usage"]["completion_tokens"])

    def test_04b_openai_metadata_reports_the_first_choice_when_n_is_greater_than_one(
        self,
    ):
        """With n > 1 the single metadata block describes the first choice instead of going missing."""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
                "temperature": 0.8,
                "n": 2,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["choices"]), 2)
        metadata = data["metadata"]
        spans = metadata["weight_versions"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], metadata["weight_version"])
        self.assertEqual(metadata["weight_version"], self._current_version())
        self.assertEqual(spans[0]["start"], 0)

    def test_05_aborted_retracted_requests_report_spans(self):
        """Requests aborted while retracted in the waiting queue still report their spans."""

        def abort_all():
            requests.post(
                f"{self.base_url}/abort_request",
                json={"abort_all": True},
                timeout=30,
            ).raise_for_status()

        results = self._run_while_paused(num_requests=4, while_paused=abort_all)

        aborted_with_spans = 0
        for data in results:
            meta_info = data["meta_info"]
            if meta_info["finish_reason"]["type"] != "abort":
                continue
            if "weight_versions" not in meta_info:
                continue
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            aborted_with_spans += 1

        self.assertGreater(
            aborted_with_spans,
            0,
            "No aborted request carried weight_versions -- the AbortReq path lost the spans.",
        )

    def test_06_running_requests_split_without_abort(self):
        """A version bump with abort_all_requests=False splits requests still in the running batch."""
        previous_version = self._current_version()

        results = self._run_while_paused(
            num_requests=4,
            while_paused=lambda: self._set_weight_version("inplace-v3"),
            mode="in_place",
            max_new_tokens=1024,
        )
        self.assertEqual(self._current_version(), "inplace-v3")

        split_count = 0
        for data in results:
            meta_info = data["meta_info"]
            self.assertNotEqual(meta_info["finish_reason"]["type"], "abort")
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            self.assertEqual(spans[0]["version"], previous_version)
            if len(spans) > 1:
                split_count += 1
                self.assertEqual(
                    [span["version"] for span in spans],
                    [previous_version, "inplace-v3"],
                )
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            split_count,
            0,
            "No running request was split -- the running batch was not visited.",
        )

    def test_07_update_without_weight_version_does_not_split(self):
        """A refit that carries no weight_version leaves attribution untouched."""
        version = self._current_version()

        results = self._run_while_paused(
            num_requests=4, while_paused=self._update_weights_from_disk
        )
        self.assertEqual(self._current_version(), version)

        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["version"], version)
            self.assertEqual(spans[0]["end"], meta_info["completion_tokens"])

    def test_08_reannouncing_current_version_is_a_noop(self):
        """Re-announcing the version the server already has must not split anything."""
        version = self._current_version()

        results = self._run_while_paused(
            num_requests=4,
            while_paused=lambda: self._set_weight_version(version),
            mode="in_place",
            max_new_tokens=128,
        )
        self.assertEqual(self._current_version(), version)

        for data in results:
            spans = _assert_spans_contiguous(self, data["meta_info"])
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["version"], version)

    def test_09_three_spans_across_two_updates(self):
        """Two updates during one request produce three ordered, non-empty spans."""
        first_version = self._current_version()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._generate,
                    max_new_tokens=2048,
                    prompt=f"Write a long story about the number {i}.",
                )
                for i in range(4)
            ]

            for new_version in ("multi-a", "multi-b"):
                time.sleep(2)
                self._pause("in_place")
                try:
                    self._set_weight_version(new_version)
                finally:
                    self._continue()

            results = [future.result() for future in futures]

        expected = [first_version, "multi-a", "multi-b"]
        three_span_count = 0
        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            versions = [span["version"] for span in spans]
            self.assertEqual(versions, expected[: len(versions)])
            for span in spans:
                self.assertGreater(span["end"], span["start"])
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            if len(spans) == 3:
                three_span_count += 1

        self.assertGreater(
            three_span_count,
            0,
            "No request spanned both updates -- multi-event accumulation was not exercised.",
        )

    def test_10_abort_before_first_token_reports_empty_span(self):
        """A request aborted before producing a token still reports a well-formed span."""
        version = self._current_version()

        def abort_all():
            requests.post(
                f"{self.base_url}/abort_request",
                json={"abort_all": True},
                timeout=30,
            ).raise_for_status()

        results = self._run_while_paused(
            num_requests=16,
            while_paused=abort_all,
        )

        empty_aborts = [
            data["meta_info"]
            for data in results
            if data["meta_info"]["finish_reason"]["type"] == "abort"
            and data["meta_info"]["completion_tokens"] == 0
        ]
        self.assertGreater(len(empty_aborts), 0)
        for meta_info in empty_aborts:
            self.assertEqual(
                meta_info["weight_versions"],
                [{"version": version, "start": 0, "end": 0}],
            )
            self.assertEqual(meta_info["weight_version"], version)

    def test_11_streaming_reports_spans_only_on_the_final_chunk(self):
        """Intermediate stream chunks carry no spans; the finishing chunk carries them all."""
        max_new_tokens = 32
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "stream": True,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)

        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            chunks.append(json.loads(payload))

        version = self._current_version()
        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertNotIn("weight_versions", chunk["meta_info"])
            self.assertEqual(chunk["meta_info"]["weight_version"], version)

        meta_info = chunks[-1]["meta_info"]
        spans = _assert_spans_contiguous(self, meta_info)
        self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
        self.assertEqual(spans[-1]["end"], max_new_tokens)

    def test_12_completions_endpoint_reports_metadata(self):
        """/v1/completions surfaces the spans the same way /v1/chat/completions does."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        metadata = data["metadata"]
        self.assertEqual(metadata["weight_version"], self._current_version())
        spans = metadata["weight_versions"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], metadata["weight_version"])
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[0]["end"], data["usage"]["completion_tokens"])

    def test_13_new_requests_after_the_endpoint_returns_see_the_new_version(self):
        """Once the endpoint returns, every concurrently admitted request stamps the new version."""
        self._set_weight_version("dp-v1")
        self.assertEqual(self._current_version(), "dp-v1")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._generate, max_new_tokens=8) for _ in range(8)
            ]
            results = [future.result() for future in futures]

        for data in results:
            spans = _assert_spans_contiguous(self, data["meta_info"])
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["version"], "dp-v1")

    def test_14_all_inflight_requests_are_swept(self):
        """A version change must split every in-flight request, not just the one the sweep happens to reach first."""
        previous_version = self._current_version()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    self._generate,
                    max_new_tokens=1024 - 16 * i,
                    prompt=f"Write a long story about the number {i}.",
                )
                for i in range(8)
            ]
            time.sleep(2)
            self._set_weight_version("dp-v2")
            results = [future.result() for future in futures]

        split_count = 0
        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            self.assertEqual(spans[0]["version"], previous_version)
            if len(spans) > 1:
                split_count += 1
                self.assertEqual(
                    [span["version"] for span in spans],
                    [previous_version, "dp-v2"],
                )
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            split_count,
            1,
            "At most one in-flight request was split -- an attention-DP rank was "
            "likely missed by the weight version sweep.",
        )

    def test_15_single_span_covers_all_accepted_tokens(self):
        """Draft-token overshoot must not leak past the reported completion length."""
        data = self._generate(max_new_tokens=32)

        meta_info = data["meta_info"]
        spans = _assert_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], self._current_version())
        self.assertEqual(spans[0]["end"], meta_info["completion_tokens"])

    def test_16_retract_update_continue_keeps_exact_boundaries(self):
        """Spans stay contiguous and clamped when a retract-paused update lands mid-speculation."""
        previous_version = self._current_version()
        results = self._run_while_paused(
            num_requests=4,
            while_paused=lambda: self._set_weight_version("spec-v1"),
        )

        split_count = 0
        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            self.assertEqual(spans[0]["version"], previous_version)
            if len(spans) > 1:
                split_count += 1
                self.assertEqual(
                    [span["version"] for span in spans],
                    [previous_version, "spec-v1"],
                )
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            split_count,
            0,
            "No request spanned the update -- the retract boundary under "
            "speculative decoding was not recorded.",
        )


if __name__ == "__main__":
    unittest.main()
