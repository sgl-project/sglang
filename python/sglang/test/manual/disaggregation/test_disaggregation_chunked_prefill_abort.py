import threading
import time
import unittest
import uuid
from typing import Any

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

PD_EXTRA_ARGS = ["--max-running-requests", "4", "--chunked-prefill-size", "64"]
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


class TestDisaggChunkedPrefillAbort(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = PD_EXTRA_ARGS
        cls.extra_decode_args = PD_EXTRA_ARGS
        cls.launch_all()

    def _post_abort(self, rid: str):
        for url in (self.prefill_url, self.decode_url):
            requests.post(
                url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )

    def test_abort_mid_chunked_prefill_by_rid(self):
        rid = f"pd-chunked-prefill-abort-{uuid.uuid4().hex}"
        result: dict[str, Any] = {}

        def run_generate():
            try:
                response = requests.post(
                    self.lb_url + "/generate",
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

        time.sleep(1.0)
        abort_deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < abort_deadline:
            self._post_abort(rid)
            time.sleep(0.2)

        thread.join(timeout=60)
        self.assertFalse(thread.is_alive(), "Chunked-prefill abort request hung")
        self.assertNotIn("exception", result, result.get("exception"))
        self.assertTrue(
            _is_abort_result(result["status_code"], result["body"]),
            f"Expected chunked-prefill request to abort, got {result}",
        )

        for url in (self.lb_url, self.prefill_url, self.decode_url):
            health = requests.get(url + "/health", timeout=10)
            self.assertEqual(health.status_code, 200, health.text)


if __name__ == "__main__":
    unittest.main()
