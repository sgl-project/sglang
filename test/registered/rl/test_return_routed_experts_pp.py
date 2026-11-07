"""Routed experts from a pipeline-parallel server match a server without
pipeline parallelism for every layer, including those of later stages."""

import unittest

import test_return_routed_experts as single_stage

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")


class TestReturnRoutedExpertsPipelineParallel(single_stage.TestReturnRoutedExperts):
    """--pp-size 2 against the same server without pipeline parallelism."""

    parallel_args = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        common = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            # Arbitrary; small enough that most prompts prefill in several chunks.
            "--chunked-prefill-size",
            256,
            *cls.parallel_args,
        ]
        cls.baseline_args = common
        cls.reference_args = common + ["--pp-size", 2]

    # Not pipeline-specific; the base class already runs it.
    test_mixed_return_routed_experts_batch_alignment = None


class TestReturnRoutedExpertsPipelineParallelAttentionDP(
    TestReturnRoutedExpertsPipelineParallel
):
    """Ranks that send no outputs must still relay their layers' rows."""

    parallel_args = [
        "--tp",
        2,
        "--attn-dp-size",
        2,
        # Workaround for sgl-project/sglang#30712: capture under attention DP
        # crashes on prefill CUDA graphs; drop once that is fixed.
        "--cuda-graph-backend-prefill=disabled",
    ]


if __name__ == "__main__":
    unittest.main()
