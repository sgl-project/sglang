"""Strict log-probability correctness tests for EAGLE speculative decoding.

This test file uses the cross-mode logprob kit to verify that speculative
decoding (both v1 and v2) produces logprob values within tight tolerance
(decimal places >= 2) of non-speculative prefill scoring.

Replaces the earlier loose-tolerance checks (max_diff < 0.255 ≈ places=0)
with comprehensive artifact-level comparison using the logprob kit.

Tested artifacts:
    - output_token_logprobs / input_token_logprobs
    - output_top_logprobs  / input_top_logprobs
    - output_token_ids_logprobs / input_token_ids_logprobs
    - logprob_start_len boundary correctness
    - return_text_in_logprobs structural validation
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.logprob_kit import LogprobCrossModeMixin
from sglang.test.server_fixtures.eagle_fixture import EagleServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu")


# ---------------------------------------------------------------------------
# Eagle v1 (non-overlap) with tight logprob tolerance
# ---------------------------------------------------------------------------
class TestEagleV1LogprobCorrectness(EagleServerBase, LogprobCrossModeMixin):
    """EAGLE v1: cross-mode logprob correctness at places=2."""

    logprob_decimal_places = 2
    extra_args = ["--chunked-prefill-size", 128, "--max-running-requests", 8]


# ---------------------------------------------------------------------------
# Eagle v2 (overlap / spec-v2) with tight logprob tolerance
# ---------------------------------------------------------------------------
class TestEagleV2LogprobCorrectness(CustomTestCase, LogprobCrossModeMixin):
    """EAGLE v2 (spec-v2): cross-mode logprob correctness at places=2."""

    logprob_decimal_places = 2
    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model",
            cls.draft_model,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "6",
            "--page-size",
            "1",
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            "8",
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_SPEC_NAN_DETECTION.override(
            True
        ), envs.SGLANG_SPEC_OOB_DETECTION.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
