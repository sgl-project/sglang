import time
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import assert_process_healthy
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class PDLogprobParityMixin:
    # Mix in before the PD server fixture, which owns the P/D launches.
    reference_parallel_args = []
    baseline_args = []
    parity_prompts = [[1] + [100 + i % 1000 for i in range(256)]]
    parity_max_new_tokens = 4
    parity_logprob_delta = 0.05
    # Prompts longer than this are resent after flushing only the decode side,
    # so prefill serves them from its radix cache; None skips the resend.
    parity_cached_prefix_min_prompt_tokens: Optional[int] = None

    @classmethod
    def _generate(cls, *, base_url, input_ids):
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": cls.parity_max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["meta_info"]

    @staticmethod
    def _flush_cache(base_url):
        response = requests.post(
            base_url + "/flush_cache", params={"timeout": 30}, timeout=120
        )
        response.raise_for_status()

    def _assert_logprob_parity(self, *, reference, actual, label):
        reference_logprobs = reference["output_token_logprobs"]
        actual_logprobs = actual["output_token_logprobs"]
        self.assertEqual(
            [item[1] for item in reference_logprobs],
            [item[1] for item in actual_logprobs],
            label,
        )
        self.assertEqual(len(reference_logprobs), self.parity_max_new_tokens, label)
        for reference_item, actual_item in zip(reference_logprobs, actual_logprobs):
            self.assertAlmostEqual(
                reference_item[0],
                actual_item[0],
                delta=self.parity_logprob_delta,
                msg=label,
            )

    def _collect_references(self):
        baseline = popen_launch_server(
            self.model,
            self.lb_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.reference_parallel_args
            + ["--trust-remote-code"]
            + self.baseline_args,
            env=self.extra_prefill_env,
        )
        try:
            references = []
            for prompt in self.parity_prompts:
                self._flush_cache(self.lb_url)
                references.append(
                    self._generate(base_url=self.lb_url, input_ids=prompt)
                )
        finally:
            kill_process_tree(baseline.pid, wait_timeout=60)
        time.sleep(5)
        return references

    def _check_prompt_parity(self, *, prompt, reference):
        self._flush_cache(self.prefill_url)
        self._flush_cache(self.decode_url)
        actual = self._generate(base_url=self.lb_url, input_ids=prompt)
        self._assert_logprob_parity(
            reference=reference, actual=actual, label=f"prompt_tokens={len(prompt)}"
        )
        min_tokens = self.parity_cached_prefix_min_prompt_tokens
        if min_tokens is None or len(prompt) <= min_tokens:
            return
        self._flush_cache(self.decode_url)
        cached = self._generate(base_url=self.lb_url, input_ids=prompt)
        self.assertGreater(cached["cached_tokens"], 0)
        self._assert_logprob_parity(
            reference=reference,
            actual=cached,
            label=f"cached prompt_tokens={len(prompt)}",
        )

    def test_logprob_parity(self):
        references = self._collect_references()
        self.launch_all()
        for prompt, reference in zip(self.parity_prompts, references):
            with self.subTest(prompt_tokens=len(prompt)):
                self._check_prompt_parity(prompt=prompt, reference=reference)

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
