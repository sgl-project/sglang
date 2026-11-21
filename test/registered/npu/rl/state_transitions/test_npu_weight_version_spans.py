"""Phase 3 NPU tests: weight-version attribution of RL rollouts.

A token generated after a weight sync must be attributable to the version that
produced it, otherwise the trainer cannot tell which policy a rollout came from.
These cases cover the three boundaries that matter on NPU: a request that
spans an update, a prompt whose prefix was cached under an older version, and
a request aborted while the engine is paused.
"""

import os
import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    abort_all_requests,
    continue_generation,
    current_weight_version,
    flush_cache,
    npu_rl_env,
    npu_server,
    pause_generation,
    start_stream,
    update_weight_version,
    wait_for_first_token,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=900, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=900, suite="nightly-test-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
HAVE_MODEL = os.path.isfile(os.path.join(MODEL, "config.json"))
BASE_VERSION = "base-v0"
SERVER_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--context-length",
    "8192",
    "--max-running-requests",
    "8",
    "--mem-fraction-static",
    "0.55",
    "--weight-version",
    BASE_VERSION,
    # Radix reuse across versions is part of what these cases assert.
    "--enable-prefill-weight-versions",
)
# Ascend caches KV in 128-token pages, so a prefix shorter than one page is
# never reusable. Repeat the preamble until several whole pages are shared.
SHARED_PREFIX_SENTENCE = (
    "You are a meticulous assistant. Follow every instruction to the letter, "
    "explain your reasoning step by step, and never skip an intermediate result. "
)
SHARED_PREFIX = SHARED_PREFIX_SENTENCE * 40
LONG_PROMPT = "Write a long story about the number {}."


def _spans(meta_info, key="weight_versions"):
    spans = meta_info.get(key)
    if not spans:
        raise AssertionError(f"meta_info carries no {key}: {meta_info}")
    return spans


def _assert_contiguous(test, spans, *, label):
    test.assertGreater(len(spans), 0, f"{label}: empty span list")
    test.assertEqual(spans[0]["start"], 0, f"{label}: spans do not start at 0")
    for previous, current in zip(spans, spans[1:]):
        test.assertEqual(
            previous["end"], current["start"], f"{label}: spans are not contiguous"
        )
        test.assertNotEqual(
            previous["version"], current["version"], f"{label}: duplicate versions"
        )
    return spans


@unittest.skipUnless(HAVE_MODEL, f"model not found: {MODEL}")
class TestNpuWeightVersionSpans(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=SERVER_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def tearDown(self):
        try:
            continue_generation(self.base_url)
        except Exception:
            pass

    def _stream_all(self, count, *, max_new_tokens, prompt=LONG_PROMPT):
        requests = [
            start_stream(
                self.base_url,
                prompt.format(index),
                max_new_tokens=max_new_tokens - 32 * index,
            )
            for index in range(count)
        ]
        # A first token proves the request is in the running batch, which is the
        # state the version boundary has to split.
        for index, request in enumerate(requests):
            wait_for_first_token(request, label=f"request {index}")
        return requests

    def _drain(self, requests, *, timeout=240):
        results = []
        for index, request in enumerate(requests):
            request.thread.join(timeout=timeout)
            self.assertFalse(
                request.thread.is_alive(), f"request {index} did not finish"
            )
            self.assertFalse(
                request.errors, f"request {index} failed: {request.errors}"
            )
            results.append(request.events[-1])
        return results

    def test_01_boundary_splits_running_requests(self):
        base = current_weight_version(self.base_url)
        self.assertEqual(
            base, BASE_VERSION, "server started with an unexpected version"
        )

        requests = self._stream_all(4, max_new_tokens=512)
        pause_generation(self.base_url, mode="in_place")
        try:
            update_weight_version(self.base_url, "span-v1", abort_all_requests=False)
        finally:
            continue_generation(self.base_url)
        results = self._drain(requests)
        self.assertEqual(current_weight_version(self.base_url), "span-v1")

        split = 0
        for index, data in enumerate(results):
            meta_info = data["meta_info"]
            self.assertNotEqual(
                meta_info["finish_reason"]["type"], "abort", f"request {index} aborted"
            )
            spans = _assert_contiguous(
                self, _spans(meta_info), label=f"request {index} output spans"
            )
            self.assertEqual(
                spans[-1]["end"],
                meta_info["completion_tokens"],
                f"request {index}: the last span does not end at the completion length",
            )
            versions = [span["version"] for span in spans]
            self.assertEqual(versions[0], base, f"request {index}: wrong first version")
            if len(spans) > 1:
                split += 1
                self.assertEqual(versions, [base, "span-v1"])
                self.assertGreater(
                    spans[0]["end"], 0, f"request {index}: empty pre-update span"
                )

        self.assertGreater(
            split,
            0,
            "no running request was split by the update -- the boundary was not "
            "recorded for the running batch",
        )

    def test_02_reused_prefix_keeps_the_version_that_computed_it(self):
        flush_cache(self.base_url)
        old_version = current_weight_version(self.base_url)

        first_prompt = SHARED_PREFIX + "Name three colors."
        self._drain([start_stream(self.base_url, first_prompt, max_new_tokens=8)])

        update_weight_version(self.base_url, "reuse-v1", abort_all_requests=False)
        second = start_stream(
            self.base_url,
            SHARED_PREFIX + "Name three colors. Then name three shapes.",
            max_new_tokens=8,
        )
        meta_info = self._drain([second])[0]["meta_info"]

        self.assertGreater(
            meta_info["cached_tokens"],
            0,
            "the prompt prefix was not reused -- the radix cache was not hit",
        )
        prefill = _assert_contiguous(
            self, _spans(meta_info, "prefill_weight_versions"), label="prefill spans"
        )
        self.assertEqual(
            [span["version"] for span in prefill],
            [old_version, "reuse-v1"],
            "the reused prefix is not attributed to the version that computed it",
        )
        self.assertEqual(
            prefill[0]["end"],
            meta_info["cached_tokens"],
            "the stale span does not end where the reused prefix ends",
        )
        output = _assert_contiguous(self, _spans(meta_info), label="output spans")
        self.assertEqual(
            [span["version"] for span in output],
            ["reuse-v1"],
            "output tokens were attributed to the previous version",
        )

    def test_03_aborted_requests_keep_their_spans(self):
        base = current_weight_version(self.base_url)
        requests = self._stream_all(4, max_new_tokens=512)

        pause_generation(self.base_url, mode="in_place")
        try:
            update_weight_version(self.base_url, "abort-v1", abort_all_requests=False)
            abort_all_requests(self.base_url)
        finally:
            continue_generation(self.base_url)
        results = self._drain(requests)

        aborted_with_spans = 0
        for index, data in enumerate(results):
            meta_info = data["meta_info"]
            if meta_info["finish_reason"]["type"] != "abort":
                continue
            if "weight_versions" not in meta_info:
                continue
            spans = _assert_contiguous(
                self, _spans(meta_info), label=f"aborted request {index} spans"
            )
            self.assertEqual(
                spans[-1]["end"],
                meta_info["completion_tokens"],
                f"aborted request {index}: spans do not cover its tokens",
            )
            self.assertEqual(spans[0]["version"], base)
            aborted_with_spans += 1

        self.assertGreater(
            aborted_with_spans,
            0,
            "no aborted request carried weight_versions -- the abort path dropped "
            "the spans",
        )


if __name__ == "__main__":
    unittest.main()
