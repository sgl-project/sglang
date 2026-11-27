"""Phase 2 NPU tests for DP-local routed-expert slicing (attention TP=1)."""

import threading
import time
import unittest

import numpy as np

from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.npu_rl_routed_experts import (
    CASE,
    LONG_PROMPT,
    PROMPTS,
    REQUEST_TIMEOUT,
    RoutedExpertsTestMixin,
    generate,
    responses,
    seqlen,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-2-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-2-npu-a3", nightly=True)


@unittest.skipUnless(weights.case_available(CASE), f"model not found: {CASE.model}")
class TestNpuReturnRoutedExperts(RoutedExpertsTestMixin, CustomTestCase):
    tp_size = 2
    moe_a2a_backend = "none"

    def test_01_shape_and_rows(self):
        """Rows = seqlen-1; shape [rows, num_layers, topk]; ids valid and unique."""
        for prompt in PROMPTS:
            body = generate(self.base_url, prompt)
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                seqlen(body) - 1,
                f"{prompt!r}: row count != seqlen-1",
            )

    def test_02_start_len_semantics(self):
        """start_len=N crops rows to [N, seqlen-1) and matches the full tail."""
        body_full = generate(self.base_url, LONG_PROMPT)
        experts_full = self._experts(body_full)
        full_length = seqlen(body_full)
        prompt_tokens = body_full["meta_info"]["prompt_tokens"]

        # Omitted start_len must behave exactly like an explicit 0.
        body_zero = generate(self.base_url, LONG_PROMPT, routed_experts_start_len=0)
        self.assertTrue(
            np.array_equal(experts_full, self._experts(body_zero)),
            "default and explicit start_len=0 must be identical",
        )

        # start_len = k crops the leading rows; the rest is the full tail.
        k = prompt_tokens // 2
        body_k = generate(self.base_url, LONG_PROMPT, routed_experts_start_len=k)
        experts_k = self._experts(body_k)
        self.assertEqual(
            experts_k.shape[0], full_length - 1 - k, "start_len did not crop rows"
        )
        self.assertTrue(
            np.array_equal(experts_full[k:], experts_k),
            "cropped response is not the tail of the full response",
        )

        # start_len = prompt_tokens is the inclusive upper bound.
        body_prompt = generate(
            self.base_url, LONG_PROMPT, routed_experts_start_len=prompt_tokens
        )
        self.assertEqual(
            self._experts(body_prompt).shape[0],
            full_length - 1 - prompt_tokens,
            "start_len=prompt_tokens must crop to the generated tokens",
        )

    def test_03_batch_alignment_and_disabled_requests(self):
        """Every request of a batch gets its own rows; disabled requests get none.

        Also the negative path: a request that does not ask for routed experts
        must not receive the field, even when it races one that does.
        """
        batch = list(PROMPTS) + [LONG_PROMPT * 3]
        bodies = responses(generate(self.base_url, batch))
        self.assertEqual(len(bodies), len(batch), "batch reply lost a request")
        for prompt, body in zip(batch, bodies):
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                seqlen(body) - 1,
                f"rows of one request leaked across the batch for {prompt!r}",
            )

        # The same prompt twice in one greedy batch must route identically.
        twins = responses(generate(self.base_url, [PROMPTS[0], PROMPTS[0]]))
        self.assertTrue(
            np.array_equal(self._experts(twins[0]), self._experts(twins[1])),
            "identical prompts in one batch returned different routings",
        )

        results = {}
        errors = {}

        def _worker(key, return_flag):
            try:
                results[key] = generate(
                    self.base_url, PROMPTS[key], return_routed_experts=return_flag
                )
            except Exception as exc:
                errors[key] = exc

        threads = [
            threading.Thread(target=_worker, args=(0, False), daemon=True),
            threading.Thread(target=_worker, args=(1, True), daemon=True),
        ]
        deadline = time.monotonic() + REQUEST_TIMEOUT + 10
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        pending = [index for index, thread in enumerate(threads) if thread.is_alive()]
        errors = errors.copy()
        if errors:
            raise AssertionError(
                f"concurrent requests failed: {errors!r}; still pending: {pending}"
            ) from next(iter(errors.values()))
        self.assertFalse(pending, f"concurrent requests timed out: {pending}")

        self.assertNotIn(
            "routed_experts",
            results[0].get("meta_info", {}),
            "return_routed_experts=False must not return the field",
        )
        self.assertIn(
            "routed_experts",
            results[1].get("meta_info", {}),
            "return_routed_experts=True must return the field",
        )


if __name__ == "__main__":
    unittest.main()
