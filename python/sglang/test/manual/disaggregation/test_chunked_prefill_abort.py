import threading
import time
import unittest
import uuid
from typing import Any

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

CHUNKED_PREFILL_SIZE = 64
LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
) * 900


def _decode_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _is_abort_result(status_code: int, body: Any) -> bool:
    if status_code == 200:
        reason = (
            body.get("meta_info", {}).get("finish_reason", {})
            if isinstance(body, dict)
            else {}
        )
        return isinstance(reason, dict) and reason.get("type") == "abort"

    if status_code not in (500, 503):
        return False

    text = body if isinstance(body, str) else str(body)
    return "abort" in text.lower()


class TestChunkedPrefillAbortE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-cuda-graph",
                "--chunked-prefill-size",
                str(CHUNKED_PREFILL_SIZE),
                "--max-running-requests",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def test_abort_mid_chunked_prefill_by_rid(self):
        rid = f"chunked-prefill-abort-{uuid.uuid4().hex}"
        result: dict[str, Any] = {}

        def run_generate():
            try:
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "rid": rid,
                        "text": f"{rid}\n{LONG_PROMPT}",
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

        time.sleep(0.5)
        abort_deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < abort_deadline:
            requests.post(
                self.base_url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )
            time.sleep(0.2)

        thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "Chunked-prefill abort request hung")
        self.assertNotIn("exception", result, result.get("exception"))
        self.assertTrue(
            _is_abort_result(result["status_code"], result["body"]),
            f"Expected chunked-prefill request to abort, got {result}",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200, health.text)

        follow_up = requests.post(
            self.base_url + "/generate",
            json={
                "rid": f"chunked-prefill-after-abort-{uuid.uuid4().hex}",
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=60,
        )
        follow_up_body = _decode_response(follow_up)
        self.assertEqual(follow_up.status_code, 200, follow_up_body)
        self.assertFalse(_is_abort_result(follow_up.status_code, follow_up_body))


if __name__ == "__main__":
    unittest.main()
