"""MiMo V2.5 under ``--enable-unified-memory`` — the STRIDED fallback arm.

MiMo V2.5 is the in-tree asymmetric-K/V MHA model (head_dim 192 != v_head_dim
128), so its KV sub-pool rows are not uniform and the dense-view flip stays
OFF: the pool keeps envelope-strided 4-D views, the backend allow-list narrows to
Triton, and the boot log reports ``strided views``. This is the one arm of the
three-way gate no dense-view cell can reach — without it, a regression that
breaks the strided fallback (the arm every asymmetric model lives on) would
pass the whole dense matrix.

Nightly H200-only, mirroring the other MiMo e2e coverage (the checkpoint is
pre-cached on the eight-H200 runner). Spec stays off: unified memory asserts
it.

    python -m pytest test/registered/page_major/test_unified_mimo_strided_arm.py -v
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=1200, stage="nightly", runner_config="8-gpu-h200")

MIMO_MODEL = "XiaomiMiMo/MiMo-V2.5"


class TestUnifiedMiMoStridedArm(DefaultServerBase):
    """Asymmetric-K/V model on the unified pool: strided views, Triton-only."""

    model = MIMO_MODEL

    gsm8k_threshold = 0.85
    num_gsm8k_questions = 200
    num_shots = 5
    parallel = 32

    other_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--enable-unified-memory",
        # The only backend the asymmetric (strided) arm allows; pinning it
        # also documents the arm this cell exists to exercise.
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.65",
        "--cuda-graph-backend-prefill=disabled",
    ]

    def test_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=self.num_shots,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=512,
            parallel=self.parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)


if __name__ == "__main__":
    unittest.main()
