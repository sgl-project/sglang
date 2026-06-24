import json
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any

import requests

from sglang.srt.environ import envs
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

FAILURE_PROB = 0.12
PD_EXTRA_ARGS = [
    "--max-running-requests",
    "4",
    "--chunked-prefill-size",
    "128",
]
PD_LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
) * 240


def _body_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, sort_keys=True)


def _decode_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _finish_reason(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    reason = body.get("meta_info", {}).get("finish_reason", {})
    return reason if isinstance(reason, dict) else {}


def _is_abort_result(status_code: int, body: Any) -> bool:
    if status_code == 200:
        return _finish_reason(body).get("type") == "abort"

    if status_code not in (500, 503):
        return False

    text = _body_text(body).lower()
    return any(
        marker in text
        for marker in (
            "pd peer failed",
            "prefill bootstrap failed",
            "abort",
            "aborted",
        )
    )


def _is_success_result(status_code: int, body: Any) -> bool:
    return (
        status_code == 200
        and isinstance(body, dict)
        and "text" in body
        and _finish_reason(body).get("type") != "abort"
    )


class TestDisaggregationPeerLivenessAbort(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._disagg_failure_ctx = envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(
            FAILURE_PROB
        )
        cls._disagg_failure_ctx.__enter__()

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = PD_EXTRA_ARGS
        cls.extra_decode_args = PD_EXTRA_ARGS
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if cls._disagg_failure_ctx:
                cls._disagg_failure_ctx.__exit__(None, None, None)

    def _post_generate(self, rid: str, max_new_tokens: int = 32) -> dict[str, Any]:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "rid": rid,
                "text": f"{rid}\n{PD_LONG_PROMPT}",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=75,
        )
        return {
            "rid": rid,
            "status_code": response.status_code,
            "body": _decode_response(response),
        }

    def _assert_servers_healthy(self):
        for url in (self.lb_url, self.prefill_url, self.decode_url):
            response = requests.get(url + "/health", timeout=10)
            self.assertEqual(response.status_code, 200, response.text)

    def _load_count(self, url: str) -> int:
        response = requests.get(
            url + "/v1/loads?include=core,disagg,queues",
            timeout=10,
        )
        response.raise_for_status()
        total = 0
        for load in response.json()["loads"]:
            total += int(load.get("num_running_reqs", 0))
            total += int(load.get("num_waiting_reqs", 0))
            disagg = load.get("disaggregation", {})
            total += int(disagg.get("prefill_bootstrap_queue_reqs", 0))
            total += int(disagg.get("prefill_inflight_queue_reqs", 0))
            total += int(disagg.get("decode_prealloc_queue_reqs", 0))
            total += int(disagg.get("decode_transfer_queue_reqs", 0))
            total += int(disagg.get("decode_retracted_queue_reqs", 0))
            queues = load.get("queues", {})
            total += int(queues.get("waiting", 0))
            total += int(queues.get("grammar", 0))
            total += int(queues.get("paused", 0))
            total += int(queues.get("retracted", 0))
        return total

    def _assert_eventually_idle(self):
        deadline = time.monotonic() + 30
        last_counts = {}
        while time.monotonic() < deadline:
            last_counts = {
                self.prefill_url: self._load_count(self.prefill_url),
                self.decode_url: self._load_count(self.decode_url),
            }
            if all(count == 0 for count in last_counts.values()):
                return
            time.sleep(1)

        self.fail(f"PD servers still have request load after cleanup: {last_counts}")

    def _assert_one_successful_generate_with_retries(self):
        last_result = None
        for _ in range(20):
            rid = f"pd-peer-liveness-health-{uuid.uuid4().hex}"
            last_result = self._post_generate(rid, max_new_tokens=1)
            if _is_success_result(last_result["status_code"], last_result["body"]):
                return
            time.sleep(0.2)

        self.fail(
            "Server stayed HTTP-healthy but did not complete a follow-up generate "
            f"under failure injection. Last result: {last_result}"
        )

    def test_peer_failures_abort_without_hanging(self):
        num_requests = 36
        start = time.monotonic()
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=18) as executor:
            futures = [
                executor.submit(
                    self._post_generate,
                    f"pd-peer-liveness-{i}-{uuid.uuid4().hex}",
                )
                for i in range(num_requests)
            ]

            try:
                for future in as_completed(futures, timeout=120):
                    results.append(future.result())
            except TimeoutError:
                unfinished = sum(not future.done() for future in futures)
                self.fail(f"{unfinished} generate requests did not return in time")

        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 120)
        self.assertEqual(len(results), num_requests)

        aborts = [
            result
            for result in results
            if _is_abort_result(result["status_code"], result["body"])
        ]
        successes = [
            result
            for result in results
            if _is_success_result(result["status_code"], result["body"])
        ]
        unexpected = [
            result
            for result in results
            if result not in aborts and result not in successes
        ]

        self.assertFalse(
            unexpected,
            "Expected only clean aborts or successful completions. "
            f"Unexpected results: {unexpected[:3]}",
        )
        self.assertGreater(
            len(aborts),
            0,
            "Injected KVPoll.Failed should abort at least one request",
        )

        self._assert_servers_healthy()
        self._assert_eventually_idle()
        self._assert_one_successful_generate_with_retries()


class TestDisaggregationPeerLivenessCleanRecovery(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = PD_EXTRA_ARGS
        cls.extra_decode_args = PD_EXTRA_ARGS
        cls.launch_all()

    def _post_abort_to_engines(self, rid: str):
        for url in (self.prefill_url, self.decode_url):
            response = requests.post(
                url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )
            self.assertEqual(response.status_code, 200, response.text)

    def _assert_servers_healthy(self):
        for url in (self.lb_url, self.prefill_url, self.decode_url):
            response = requests.get(url + "/health", timeout=10)
            self.assertEqual(response.status_code, 200, response.text)

    def test_clean_generate_and_explicit_abort_by_rid(self):
        clean_response = requests.post(
            self.lb_url + "/generate",
            json={
                "rid": f"pd-clean-generate-{uuid.uuid4().hex}",
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=60,
        )
        clean_body = _decode_response(clean_response)
        self.assertTrue(
            _is_success_result(clean_response.status_code, clean_body), clean_body
        )

        rid = f"pd-explicit-abort-{uuid.uuid4().hex}"
        result: dict[str, Any] = {}

        def run_generate():
            try:
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "rid": rid,
                        "text": f"{rid}\n{PD_LONG_PROMPT}",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 4096,
                            "ignore_eos": True,
                        },
                    },
                    timeout=180,
                )
                result["status_code"] = response.status_code
                result["body"] = _decode_response(response)
            except requests.RequestException as exc:
                result["exception"] = repr(exc)

        thread = threading.Thread(target=run_generate)
        thread.start()

        time.sleep(0.25)
        abort_deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < abort_deadline:
            self._post_abort_to_engines(rid)
            time.sleep(0.25)

        thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "Explicitly aborted PD request hung")
        self.assertNotIn("exception", result, result.get("exception"))
        self.assertTrue(
            _is_abort_result(result["status_code"], result["body"]),
            f"Expected explicit abort result, got {result}",
        )

        self._assert_servers_healthy()


if __name__ == "__main__":
    unittest.main()
