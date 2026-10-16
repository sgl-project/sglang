"""The only test in the tree that bounds speculative decoding LATENCY; every
other one bounds accept length. CUDA only -- AMD bounds are unmeasured.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, at_most, check_perf
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=145, stage="extra-a", runner_config="1-gpu-large")


class TestEagle3Latency(CustomTestCase):
    def test_online_latency_eagle3(self):
        res = run_bench_serving(
            model=DEFAULT_TARGET_MODEL_EAGLE3,
            num_prompts=300,
            request_rate=8,
            sharegpt_context_len=3072,
            disable_ignore_eos=True,
            dataset_name="sharegpt",
            other_server_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_EAGLE3,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "16",
                "--mem-fraction-static",
                "0.7",
                # The draft checkpoint ships fp16 and the target bf16; the CUDA
                # rmsnorm path rejects a weight and activation pair that disagree.
                "--dtype",
                "float16",
            ],
            need_warmup=True,
            seed=42,
        )

        check_perf(
            self,
            at_most(
                "median_e2e_latency_ms", res["median_e2e_latency_ms"], 1150, unit="ms"
            ),
            at_least("accept_length", res["accept_length"], 2.3),
        )


if __name__ == "__main__":
    unittest.main()
