"""End-to-end coverage for --disaggregation-enable-kv-checksum.

The unit tests drive fake pools; only a real transfer shows that the two
engines digest the same bytes. TP2 on both sides (prefill on GPUs 0-1, decode
on 2-3) is deliberate: at TP1 the mismatch all-reduce is a no-op, so the hang
it prevents -- one rank dropping a request its peers keep -- never comes up.

Mismatches are injected from a per-rank generator (the global `random` is
seeded identically on every rank), so some requests abort on one rank only.
Healthy requests must still complete (no false positives), injected ones must
abort (detection), and both engines must stay up (the ranks agreed on what to
drop).
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=420, stage="base-c", runner_config="4-gpu-h100")

# Per decode rank; a request survives when neither of the two ranks injects.
_MISMATCH_PROB = 0.3
_NUM_REQUESTS = 16
_NUM_RADIX_REQUESTS = 6
# Longer than the chunk size, so the digest spans a multi-chunk handoff.
_PROMPT = "Count from one to twenty, then explain why you stopped: " * 20


def _is_abort_result(response: requests.Response) -> bool:
    """A 200 whose finish reason is "abort", or the 500 that
    _handle_kv_checksum_mismatch raises with its own message (no "abort" in
    the body, so the chunked-prefill test's predicate does not apply)."""
    if response.status_code == 200:
        reason = response.json().get("meta_info", {}).get("finish_reason", {})
        return isinstance(reason, dict) and reason.get("type") == "abort"
    return (
        response.status_code == 500 and "kv checksum mismatch" in response.text.lower()
    )


class TestDisaggregationKVChecksum(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--disaggregation-enable-kv-checksum",
            "--chunked-prefill-size",
            "128",
        ]
        cls.extra_decode_args = ["--disaggregation-enable-kv-checksum"]
        cls.extra_decode_env = {
            "SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB": str(_MISMATCH_PROB)
        }
        cls.launch_all()

    def test_detects_injected_mismatches_and_serves_the_rest(self):
        aborted, completed = 0, 0
        for _ in range(_NUM_REQUESTS):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": _PROMPT,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=120,
            )
            if _is_abort_result(response):
                aborted += 1
                continue
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["text"].strip(), response.text)
            completed += 1
        # P(abort) = 1 - (1 - p)^2 = 0.51 per request, so with 16 requests each
        # of these fails by chance about once in 10^5 runs.
        self.assertGreater(aborted, 0, "no injected mismatch was detected")
        self.assertGreater(completed, 0, "every healthy transfer was aborted")
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestDisaggregationKVChecksumDecodeRadixCache(PDDisaggregationServerBase):
    """Prompts sharing a long prefix hit decode's radix cache, so prefill sends
    and digests only the suffix. Those hits must verify; a digest over the
    whole prompt would abort every one of them as corruption (in CI the
    mismatch raises, so the decode would not even stay up)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--disaggregation-enable-kv-checksum",
            "--chunked-prefill-size",
            "128",
        ]
        cls.extra_decode_args = [
            "--disaggregation-enable-kv-checksum",
            "--disaggregation-decode-enable-radix-cache",
        ]
        cls.launch_all()

    def test_prefix_hits_verify(self):
        for i in range(_NUM_RADIX_REQUESTS):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": _PROMPT + f"Attempt {i}: ",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["text"].strip(), response.text)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
