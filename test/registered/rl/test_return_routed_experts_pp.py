"""Routed experts from a pipeline-parallel server match a server without
pipeline parallelism for every layer, including those of later stages."""

import unittest

import numpy as np
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

    @classmethod
    def _run_endpoint_test(cls, endpoint):
        cls._ensure_comparison_results()
        baseline = cls.baseline_results[endpoint]
        reference = cls.reference_results[endpoint]
        single_stage.check_all_experts_id_valid(reference)
        for prompt, (bsl, ref) in enumerate(zip(baseline, reference)):
            assert bsl.shape == ref.shape, (
                f"{endpoint} prompt {prompt}: {ref.shape[0]} rows, expected {bsl.shape[0]}"
            )
            # Top-k order within a layer is not part of the contract.
            differs = np.sort(bsl, axis=-1) != np.sort(ref, axis=-1)
            layers = np.flatnonzero(differs.any(axis=(0, 2))).tolist()
            assert not layers, f"{endpoint} prompt {prompt}: layers {layers} differ"

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
