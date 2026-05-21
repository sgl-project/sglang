from __future__ import annotations

import unittest
from typing import ClassVar, List

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.utils import CanaryE2EBase

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")


_GEMMA3_MODEL = "google/gemma-3-1b-it"

# Gemma 3 1B-it's HF config carries layer-typed rope params; SGLang's parser
# also needs an explicit rope_type / factor on full_attention. Passing this via
# --json-model-override-args avoids touching the model source.
_ROPE_OVERRIDE = (
    '{"rope_parameters":{'
    '"sliding_attention":{"rope_type":"default","rope_theta":10000},'
    '"full_attention":{"rope_type":"default","rope_theta":1000000,"factor":8.0}'
    "}}"
)

# DO NOT pass --disable-cuda-graph or --disable-piecewise-cuda-graph in any
# canary e2e test. The canary kernel must run inside the cuda graph alongside
# the real attn kernel; disabling the graph silently bypasses the only path
# that exercises that invariant end-to-end.

# Gemma 3 1B-it sliding_window = 512; any prompt tokenising to >512 tokens
# will force the SWA sub-pool to clip to the last window, which is the scenario
# tested by test_long_prompt_swa_window_clip.
_LONG_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 200).strip()


class _Gemma3SwaBase(CanaryE2EBase):
    model: ClassVar[str] = _GEMMA3_MODEL
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _ROPE_OVERRIDE,
    ]


class TestShortPromptFullSwaBothVerify(_Gemma3SwaBase, unittest.TestCase):
    def test_short_prompt_full_swa_both_verify(self) -> None:
        # Step 1: drive enough traffic to clear the cuda-graph warmup window.
        results = self.send_parallel_requests(n=16, max_new_tokens=32)

        # Step 2: every request must complete and /health must stay up.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()

        # Step 3: no canary violations surfaced (clean run).
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        violations = [line for line in haystack.splitlines() if "canary_kind:" in line]
        self.assertEqual(
            violations, [], f"Unexpected canary violation lines: {violations}"
        )


class TestLongPromptSwaWindowClip(_Gemma3SwaBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        "--json-model-override-args",
        _ROPE_OVERRIDE,
        "--context-length",
        "8192",
    ]

    def test_long_prompt_swa_window_clip(self) -> None:
        # Step 1: send a prompt that exceeds Gemma 3's SWA window (4096 tokens).
        # The SWA sub-pool clips verification to the last window; the test confirms
        # the clip path completes without raising a canary violation.
        self.assertFalse(
            self.launch_failed,
            f"Server failed to launch: {self.launch_exception!r}",
        )
        results = self.send_parallel_requests(
            n=4,
            prompts=[_LONG_PROMPT],
            max_new_tokens=8,
            timeout=120.0,
        )

        # Step 2: requests must succeed; SWA window clip is transparent to clients.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()

        # Step 3: no violation emitted (correct prefill, just clipped).
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        violations = [line for line in haystack.splitlines() if "canary_kind:" in line]
        self.assertEqual(
            violations,
            [],
            f"Unexpected canary violation lines under SWA clip: {violations}",
        )


class TestMixedBatchNoViolations(_Gemma3SwaBase, unittest.TestCase):
    def test_mixed_batch_no_violations(self) -> None:
        # Step 1: fan out concurrent requests with varying prompt lengths so the
        # scheduler sees a mixed batch (some requests in FULL layers, some in SWA).
        self.assertFalse(
            self.launch_failed,
            f"Server failed to launch: {self.launch_exception!r}",
        )
        mixed_prompts = [
            "Hi",
            "The quick brown fox",
            "Explain in one sentence what a transformer is.",
            "1 + 1 =",
            "The capital of France is",
            "Once upon a time",
            "What is the meaning of life?",
            "Summarize the following text in one sentence:",
        ]
        results = self.send_parallel_requests(
            n=32, prompts=mixed_prompts, max_new_tokens=16, timeout=60.0
        )

        # Step 2: every request must complete with status 200.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)

        # Step 3: violation counter must stay at zero across the whole batch.
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        violations = [line for line in haystack.splitlines() if "canary_kind:" in line]
        self.assertEqual(
            violations,
            [],
            f"Unexpected canary violation lines in mixed batch: {violations}",
        )


if __name__ == "__main__":
    unittest.main()
