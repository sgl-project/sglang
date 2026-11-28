"""E2E gate for --enable-prefill-weight-versions on the NPU backend.

Fuller coverage (pause/update races, DP, speculative decoding) lives in
test/registered/rl/test_weight_version_spans.py.
"""

import unittest

import requests

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    admin_post,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-test-1-npu-a3", nightly=True)


class TestNpuPrefillWeightVersions(CustomTestCase):
    """E2E: --enable-prefill-weight-versions span attribution on NPU.

    [Test Category] RL Weight Versions
    [Test Target] --enable-prefill-weight-versions, POST /update_weight_version
    """

    REQUEST_TIMEOUT = 120

    # P1's tokens are a strict prefix of P2's ("." blocks tokenizer merges
    # across the boundary); the shared prefix spans several radix pages.
    _SENTENCE = "The capital of France is Paris. The capital of Germany is Berlin."
    PROMPT_P1 = " ".join([_SENTENCE] * 20)
    PROMPT_P2 = PROMPT_P1 + " The capital of Italy is Rome."

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-decode-cuda-graph",
                "--disable-prefill-cuda-graph",
                # The NPU default, explicit so S2's page-rounded cache hit
                # is deterministic.
                "--page-size",
                "128",
                "--enable-prefill-weight-versions",
                # /update_weight_version requires the admin key.
                *ADMIN_LAUNCH_ARGS_FOR_TEST,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _generate(self, text: str, max_new_tokens: int = 8):
        """Send a deterministic /generate request, return meta_info."""
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["meta_info"]

    def test_prefill_weight_versions(self):
        """
        S1: Fresh prompt → single span under the initial version.
        S2: Version relabel + prefix reuse → split spans with correct
            attribution on each side of the cache-hit boundary.
        """
        # ── S1: fresh prompt, whole prompt computed under "default" ──
        meta1 = self._generate(self.PROMPT_P1)
        spans1 = meta1["prefill_weight_versions"]
        self.assertEqual(meta1["weight_version"], "default")
        self.assertEqual(
            spans1,
            [{"version": "default", "start": 0, "end": meta1["prompt_tokens"]}],
            f"S1: fresh prompt should be a single span, got {spans1}",
        )

        # ── S2: relabel, then reuse P1's cached prefix inside P2 ──
        response = admin_post(
            self.base_url + "/update_weight_version",
            json={"new_version": "v2", "abort_all_requests": False},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)

        meta2 = self._generate(self.PROMPT_P2)
        self.assertEqual(
            meta2["cached_tokens"],
            meta1["prompt_tokens"] // 128 * 128,
            "S2: P1's prefix must hit the radix cache for the split to exist",
        )
        spans2 = meta2["prefill_weight_versions"]
        self.assertEqual(
            [span["version"] for span in spans2],
            ["default", "v2"],
            f"S2: expected old-version prefix + new-version suffix, got {spans2}",
        )
        # The boundary is the page-rounded cache-hit length, not P1's
        # prompt_tokens.
        self.assertEqual(
            spans2[0],
            {"version": "default", "start": 0, "end": meta2["cached_tokens"]},
            f"S2: reused prefix should be attributed to the old version "
            f"'default' up to the cache-hit boundary, got {spans2}",
        )
        self.assertEqual(
            spans2[-1]["end"],
            meta2["prompt_tokens"],
            f"S2: spans must cover the full prompt, got {spans2}",
        )
        # Output-side attribution rides along in the same response
        self.assertEqual(
            meta2["weight_version"], "v2", "S2: active version should be v2"
        )


if __name__ == "__main__":
    unittest.main()
