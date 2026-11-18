"""Paged experts match an unpaged TP2 server, in captured and host-planned execution."""

import io
import re
import unittest

from test_paged_experts import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    PAGED_ARGS,
    PROMPTS,
    _PagedMatchesUnpaged,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")


class TestPagedExpertsTp2(_PagedMatchesUnpaged, CustomTestCase):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE
    server_args = ("--tp-size", "2", "--moe-runner-backend", "triton")
    # Each service uses two GPUs: the reference occupies 0,1 and the paged one 2,3.
    paged_extra_args = (
        "--base-gpu-id",
        "2",
        "--log-level",
        "debug",
        "--decode-log-interval",
        "1",
    )
    prompts = [
        *PROMPTS,
        "Explain the connections in the following passage. "
        + "The ocean carries heat, rivers shape the land, and forests store carbon. "
        * 64,
    ]

    @classmethod
    def setUpClass(cls):
        cls.paged_log = io.StringIO()
        super().setUpClass()

    def test_paged_paths_are_exercised(self):
        log = self.paged_log.getvalue()
        self.assertRegex(
            log, r"Decode batch[^\n]*#running-req: 1,[^\n]*cuda graph: True"
        )
        self.assertRegex(
            log,
            r"Decode batch[^\n]*#running-req: (?:[2-9]|\d{2,}),[^\n]*cuda graph: False",
        )
        # The capture warmups use at most 128 tokens. Observe a real long
        # prefill forward with more than one wave and more than K experts.
        multiwave = re.findall(
            r"Paged experts: host-planned (\d+) waves for (\d+) tokens \((\d+) experts\)",
            log,
        )
        k = int(PAGED_ARGS[-1])
        self.assertTrue(
            any(
                int(waves) > 1 and int(tokens) >= 256 and int(experts) > k
                for waves, tokens, experts in multiwave
            ),
            f"no long prefill with multiple waves: {multiwave[-10:]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
