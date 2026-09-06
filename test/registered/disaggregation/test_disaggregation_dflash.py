"""DFLASH speculative decoding under prefill/decode (PD) disaggregation.

Mirrors the EAGLE disaggregation test in test_disaggregation_basic.py with the
DFLASH path active on both prefill and decode sides. Uses the small Llama-3.1-8B
target + its DFlash draft so CI can run it on 2 GPUs (one prefill, one decode).
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
)

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationDFlash(PDDisaggregationServerBase):
    # The DFlash draft's max_position_embeddings (40960) is shorter than the
    # Llama-3.1-8B target's (131072); allow the shorter draft context.
    extra_prefill_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    extra_decode_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_TARGET_MODEL_DFLASH
        spec_args = [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_DFLASH,
            "--speculative-dflash-block-size",
            "8",
            "--attention-backend",
            "triton",
            "--speculative-draft-attention-backend",
            "triton",
            "--cuda-graph-max-bs-decode",
            "8",
            "--mem-fraction-static",
            "0.7",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = spec_args
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.74)


if __name__ == "__main__":
    unittest.main()
