"""End-to-end coverage for --disaggregation-enable-kv-checksum.

The unit tests drive fake pools, so they cannot tell whether the two engines
agree on what bytes a handoff covers. Only a real transfer can.

Everything here runs at TP2 on both sides (prefill on GPUs 0-1, decode on
2-3). That is not incidental: at TP1 `_all_reduce_kv_checksum_mismatches`
takes its `world_size == 1` early return, so the fix for the most severe
issue -- a rank-local drop splitting the waiting queue and hanging the next
collective -- would never execute.

Two shapes, because each passes while the other is broken: healthy traffic
must not abort, and an injected fault must abort the request without hanging
the engines.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

# base-c owns the 4-gpu-h100 suite; base-b has no such runner, and
# register_cuda_ci is AST-parsed, so a bad pair fails silently.
register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-h100")

_CHECKSUM_ARGS = ["--disaggregation-enable-kv-checksum"]


def _decode_body(response: requests.Response):
    try:
        return response.json()
    except ValueError:
        return response.text


def _is_abort_result(status_code: int, body) -> bool:
    """An aborted request can surface either way.

    Same contract as `test_disaggregation_chunked_prefill_abort._is_abort_result`:
    a 200 carrying `meta_info.finish_reason.type == "abort"`, or a 5xx whose
    body names the abort. Asserting on the status code alone passes or fails
    for the wrong reason.
    """
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


class _TP2Base(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2


class TestDisaggregationKVChecksum(_TP2Base):
    """Healthy traffic: accuracy unchanged and nothing is aborted."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # Chunked prefill, so the digest spans a handoff sent as several chunks
        # and is taken on the last one.
        cls.extra_prefill_args = list(_CHECKSUM_ARGS) + [
            "--chunked-prefill-size",
            "1024",
        ]
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)

    def test_no_healthy_request_is_aborted(self):
        for prompt in (
            "The capital of France is",
            "Write a haiku about disaggregated inference:",
            "Count from one to twenty: " * 40,
        ):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = _decode_body(response)
            self.assertFalse(
                _is_abort_result(response.status_code, body),
                f"healthy transfer was aborted: {response.text}",
            )
            self.assertTrue(body.get("text"), response.text)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestDisaggregationKVChecksumCorruption(_TP2Base):
    """Corruption is detected, and a one-rank fault aborts rather than hangs.

    Without this the healthy cases all pass if `compute()` returned a
    constant. The injection is evaluated independently on each decode rank, so
    ranks routinely disagree about a given request -- exactly the split the
    all-reduce exists to prevent.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_args = list(_CHECKSUM_ARGS)
        cls.extra_decode_env = {"SGLANG_TEST_DISAGG_KV_CORRUPT_PROB": "0.5"}
        cls.launch_all()

    def test_corrupted_kv_is_rejected_without_hanging(self):
        outcomes = []
        for _ in range(8):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=120,
            )
            outcomes.append(
                _is_abort_result(response.status_code, _decode_body(response))
            )
        # Detection fires: with p=0.5 per decode rank, a request aborts with
        # probability 0.75, so P(none of 8) is about 1.5e-5.
        self.assertTrue(any(outcomes), outcomes)
        # And the engines survive it, which is what the all-reduce buys.
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
