"""triton attention backend (EAGLE3 topk=1 chain + EAGLE/Llama-2 topk=8 tree).

triton runs everywhere, so this stays on the cheap (5090) runner.

The intel_xpu attention backend mirrors the EAGLE3 topk=1 chain on the XPU
runner; the CUDA classes are gated off there and vice versa.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecHiddenStatesKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-small")
register_xpu_ci(est_time=480, stage="stage-b", runner_config="1-gpu-xpu")

_is_xpu = is_xpu()


@unittest.skipIf(_is_xpu, "CUDA runner only")
class TestEagle3Triton(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE3 spec v2 on triton (kits listed in bases)."""

    attention_backend = "triton"
    max_running_requests = 64
    cuda_graph_max_bs_decode = 64
    gsm8k_num_examples = 1000
    gsm8k_check_accept_len = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


@unittest.skipIf(_is_xpu, "CUDA runner only")
class TestEagleLlama2Triton(
    EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit, SpecHiddenStatesKit
):
    """EAGLE/Llama-2 topk=8 tree on triton.

    Hosts SpecHiddenStatesKit: topk>1 exercises the tree accept-path
    compaction that the per-req hidden-state stride slicing depends on.
    """

    attention_backend = "triton"
    enable_return_hidden_states = True
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


@unittest.skipUnless(_is_xpu, "XPU runner only")
class TestEagle3IntelXPU(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE3 spec v2 on the intel_xpu attention backend (kits listed in bases)."""

    attention_backend = "intel_xpu"
    max_running_requests = 24
    gsm8k_num_examples = 1000
    gsm8k_check_accept_len = False
    # Two XPU-specific constraints, both handled via extra_args:
    #  * --max-total-tokens: the B60 level-zero driver's mem_get_info() always
    #    reports the full 24GB as free (resident weights are invisible to it), so
    #    the KV-pool profiler over-allocates and trips UR_RESULT_ERROR_DEVICE_LOST.
    #    Hard-cap the pool (128KB/token for 8B bf16 => 16384 tokens ~= 2GB, well
    #    inside the ~8GB free after weights) instead of relying on mem-fraction.
    #  * --disable-decode-cuda-graph: XPUAttentionBackend / XPUGraphRunner reject
    #    speculative decoding during decode-graph capture, so run decode eagerly.
    extra_args = ("--max-total-tokens", "16384", "--disable-decode-cuda-graph")
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
