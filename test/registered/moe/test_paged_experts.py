"""End-to-end accuracy test for Paged Experts (srt/layers/moe/paged_experts).

Launches an MoE model with ``--enable-paged-experts`` and a forced-small resident pool so paging
actually engages (K < E), then checks GSM8K accuracy. Paging is lossless by construction (the right
experts are paged into the right slots and the routing is remapped exactly), so the score must match the
fully-resident model.

Mixtral-8x7B-Instruct (~90 GB bf16) does not fit a single H100, so ``--enable-paged-experts`` is in fact
required to serve it on this suite — the test exercises the real host->GPU paging path end to end.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")


class TestPagedExpertsAccuracy(CustomTestCase, GSM8KMixin):
    # Mixtral-8x7B has 8 experts (top_k=2); K=4 forces ~half the experts to be paged, so the test covers
    # the page-in path rather than degenerating to a fully-resident run. Lossless -> matches the
    # fully-resident Mixtral-8x7B-Instruct GSM8K score.
    gsm8k_score_threshold = 0.6

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-paged-experts",
                "--paged-experts-num-resident",
                "4",
                "--disable-cuda-graph",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # test_gsm8k is provided by GSM8KMixin


if __name__ == "__main__":
    unittest.main()
