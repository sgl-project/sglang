"""End-to-end coverage for the PD KV-cache checksum.

Two things have to hold for the check to be worth running in production:
it must not fire on healthy traffic (accuracy is unchanged, no request is
spuriously aborted), and it must fire when the KV a decode request was
handed is not the KV the prefill sent.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import DisaggKVChecksumLevel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationKVChecksum(PDDisaggregationServerBase):
    """Checksum on, healthy transfers: nothing may change but the cost."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        checksum_env = {
            "SGLANG_DISAGGREGATION_KV_CHECKSUM": str(
                int(DisaggKVChecksumLevel.SAMPLED)
            ),
        }
        cls.extra_prefill_env = dict(checksum_env)
        cls.extra_decode_env = dict(checksum_env)
        # Chunked prefill, so the digest spans a handoff sent as several chunks.
        cls.extra_prefill_args = ["--chunked-prefill-size", "1024"]
        # Decode-side radix cache, so a prefix hit makes the prefill send only
        # a suffix -- the digest then has to cover exactly the transferred
        # range on both sides, not the whole prompt.
        cls.extra_decode_args = ["--disaggregation-decode-enable-radix-cache"]
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

    def test_no_request_is_aborted(self):
        """A healthy transfer must never trip the checksum."""
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
            self.assertNotIn("corruption", response.text.lower())
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_partial_transfer_after_decode_prefix_hit(self):
        """A decode prefix hit shrinks the handoff to a suffix of the prompt.

        The digest covers only what was transferred, so the repeat -- which
        sends far fewer tokens than the first request -- must still line up.
        """
        prompt = "Summarize the following. " + "The quick brown fox jumps. " * 300
        outputs = []
        for _ in range(3):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 24},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            outputs.append(response.json()["text"])
        # Same prompt, greedy: a partial transfer must reconstruct the same KV.
        self.assertEqual(len(set(outputs)), 1, outputs)


class TestDisaggregationKVChecksumDetectsCorruption(PDDisaggregationServerBase):
    """Clobber the landed KV the way a reused slot would; the check must catch it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # SAMPLED on both sides -- the level production would run. The
        # injected fault rewrites every buffer of a row, the way a reused slot
        # does, so even a two-buffer sample sees it. The level is part of the
        # layout signature, so both sides have to agree on it.
        sampled = str(int(DisaggKVChecksumLevel.SAMPLED))
        cls.extra_prefill_env = {"SGLANG_DISAGGREGATION_KV_CHECKSUM": sampled}
        cls.extra_decode_env = {
            "SGLANG_DISAGGREGATION_KV_CHECKSUM": sampled,
            "SGLANG_TEST_DISAGG_KV_CORRUPT_PROB": "1.0",
        }
        cls.launch_all()

    def test_corrupted_kv_is_rejected(self):
        """The request fails instead of decoding against someone else's KV."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertNotEqual(
            response.status_code,
            200,
            f"corrupted KV was served as a normal completion: {response.text}",
        )
        self.assertIn("corruption", response.text.lower(), response.text)

    def test_servers_survive_the_abort(self):
        """A detected corruption aborts one request, not the engine."""
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
