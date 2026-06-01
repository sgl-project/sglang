"""topk > 1 tree drafting (EAGLE3 topk16 + EAGLE/Llama-2 topk8).

topk > 1 always routes to spec v1; flashinfer is pinned (topk > 1 can't use fa3).
Runs on the cheap (5090) runner -- functional sanity only, no perf/stress.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=840, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Topk16(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit, SpecLogprobKit):
    """EAGLE3 topk=16 tree (spec v1): correctness + gsm8k + logprob losslessness."""

    spec_topk = 16
    spec_tokens = 64
    disable_overlap = True  # topk>1 -> spec v1
    cuda_graph_max_bs = 5
    acc_length_thres = 3.1
    batch_accept_len_thres = 1.75
    gsm8k_accept_len_thres = 2.4  # EAGLE3 topk16 gsm8k accept ~2.48


class TestEagleLlama2Suite(
    EagleLlama2Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=8 full coverage (kits listed in bases)."""


class TestEagleLlama2Chunked4(EagleLlama2Base, SpecCorrectnessKit):
    """Correctness under tiny chunked prefill."""

    chunked_prefill_size = 4


class TestEagleLlama3TokenMap(EagleLlama2Base, SpecAccuracyKit):
    """EAGLE on Llama-3-8B with a FR-Spec token map (topk=4)."""

    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model = "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B"
    spec_topk = 4
    spec_tokens = 8
    cuda_graph_max_bs = 5
    gsm8k_accept_len_thres = 2.5  # FR-Spec token map lowers accept (~2.57)
    extra_args = (
        "--speculative-token-map",
        "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
    )


if __name__ == "__main__":
    unittest.main()
