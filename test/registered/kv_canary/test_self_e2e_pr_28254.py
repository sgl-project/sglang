"""Regression for PR #28254 EAGLE chunked-prefill next-prompt-token.

``_compute_chunked_req_next_prompt_token`` must index the full fill sequence
``full_untruncated_fill_ids = origin_input_ids + output_ids``, not
``origin_input_ids`` alone. A request retracted mid-decode and then resumed
re-prefills ``origin + output`` as a chunked prefill, so a non-final chunk
boundary can land in the output region. Indexing ``origin_input_ids`` returns
``None`` there, and the EAGLE prefill tail-token rotation (the only consumer of
``chunked_req_next_prompt_token``) falls back to the speculatively predicted
token instead of the already-known fill token, corrupting that chunk's draft KV.

The prompt-region boundary that ``test_self_e2e_pr_26329`` exercises cannot reach
this: the output region only exists once a request has decoded and is re-prefilled.
This test decodes a request well past the first chunk, retracts it via
``/pause_generation`` and resumes via ``/continue_generation`` so the re-prefill's
non-final boundaries land in the output region, where the kv_canary
``verify_token`` validator catches the wrong tail token. Reverting the fix
(``SGLANG_DEBUG_REVERT_PR=28254``) makes the canary fire; with the fix it is clean.
"""

from __future__ import annotations

import random
import string
import threading
import time
import unittest
from typing import ClassVar

import requests

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-small")

# Small chunk so the resumed origin+output re-prefill has many non-final
# boundaries in the output region (raises the chance the wrong tail token differs
# from the EAGLE draft prediction at >=1 boundary, which is what the canary sees).
_CHUNK = 64
_EAGLE_CHUNKED_SERVER_ARGS = (
    "--speculative-algorithm",
    "EAGLE",
    "--chunked-prefill-size",
    str(_CHUNK),
    "--cuda-graph-max-bs",
    "1",
    "--max-running-requests",
    "4",
    # Two subclasses launch their servers sequentially in one process; a small
    # model lets the second server start even if the first's GPU memory has not
    # been fully released yet (avoids a cross-subclass OOM).
    "--mem-fraction-static",
    "0.4",
)

_RID = "eagle-retract-output-region"
# Decode this long before retracting so origin+output spans many chunks.
_DECODE_SECONDS = 4.0
_RESUME_SETTLE_SECONDS = 2.0
# Sample (not greedy) so the generated output is NOT the model's own greedy
# prediction. Under greedy decoding EAGLE predicts its own output perfectly
# (accept rate 1.0), so the wrong tail token equals the real fill token and the
# bug stays invisible. With sampling, the re-prefill's freshly sampled tail
# differs from the originally sampled fill token at the chunk boundary, which is
# the divergence the canary's verify_token detects.
_TEMPERATURE = 1.0


class _EagleRetractOutputRegionBase(CanaryE2EBase):
    model_mode = "mha"
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _EAGLE_CHUNKED_SERVER_ARGS

    revert_pr: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _EagleRetractOutputRegionBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        cls.extra_env = {"SGLANG_DEBUG_REVERT_PR": "28254"} if cls.revert_pr else {}
        super().setUpClass()

    def _short_random_prompt(self) -> str:
        # ~40 tokens after BPE (< chunk), so the initial prefill is a single chunk
        # and the output region only appears on the post-retract re-prefill. Random
        # so the model's output is not trivially predictable by the EAGLE draft.
        rng = random.Random(0)
        return "".join(rng.choices(string.ascii_letters + string.digits + " ", k=120))

    def _generate_in_thread(self, result: dict) -> threading.Thread:
        def run() -> None:
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": self._short_random_prompt(),
                    "sampling_params": {
                        "max_new_tokens": 4096,
                        "temperature": _TEMPERATURE,
                        "ignore_eos": True,
                    },
                    "rid": _RID,
                    "stream": False,
                },
                timeout=120,
            )
            result["status_code"] = resp.status_code

        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def test_retract_resume_output_region(self) -> None:
        result: dict = {}
        thread = self._generate_in_thread(result)

        # Decode well past the first chunk so the accumulated output spans several
        # chunks once it is re-prefilled.
        time.sleep(_DECODE_SECONDS)
        requests.post(
            self.base_url + "/pause_generation", json={"mode": "retract"}, timeout=30
        )
        requests.post(
            self.base_url + "/continue_generation",
            json={"torch_empty_cache": False},
            timeout=30,
        )
        # Let the resumed origin+output re-prefill run; the canary verifies here.
        time.sleep(_RESUME_SETTLE_SECONDS)
        requests.post(self.base_url + "/abort_request", json={"rid": _RID}, timeout=10)
        thread.join(timeout=60)

        if self.revert_pr:
            self.assert_violation_logged_any(
                launch_tag_patterns=("*",),
                fail_reason="verify_token",
                flush_wait_seconds=3.0,
                max_retries=5,
            )
        else:
            self.assert_no_violation(wait_seconds=3.0)


class TestEagleRetractOutputRegionRegression(_EagleRetractOutputRegionBase):
    """Revert PR #28254 fix; expect a kv_canary verify_token violation."""

    revert_pr = True


class TestEagleRetractOutputRegionClean(_EagleRetractOutputRegionBase):
    """With the PR #28254 fix in place, the same retract/resume runs clean."""

    revert_pr = False


if __name__ == "__main__":
    unittest.main()
