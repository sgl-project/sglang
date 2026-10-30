"""Deferred destination KV allocation with a prefill server spread over two
GPUs: pipeline parallelism, or attention DP with expert parallelism."""

import unittest

# A module import, so the single-GPU classes are not collected again here.
import test_prefill_complete_allocation as single_gpu

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")


class _PipelineParallelPrefill:
    prefill_parallel_args = ["--pp-size", "2"]
    # After the prefill server's two GPUs.
    decode_base_gpu_id = 2


class _DpAttentionPrefill:
    """Attention DP with expert parallelism (DEP) on prefill: each DP rank is
    its own scheduler and the source of the rooms it prefills."""

    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    model_args = []
    prefill_tp_size = 2
    prefill_parallel_args = ["--attn-dp-size", "2", "--ep-size", "2"]
    decode_base_gpu_id = 2


class _AllocationCases:
    """Not a TestCase, so unittest collects these only through the classes
    below."""

    # Not a module-level name: unittest and pytest collect every TestCase
    # class bound in the module, and would rerun its single-GPU cases here.
    _cases = single_gpu.TestPrefillCompleteAllocation
    test_multi_chunk_input_logprobs = _cases.test_multi_chunk_input_logprobs
    test_concurrent_requests_complete = _cases.test_concurrent_requests_complete
    test_decode_abort_before_allocation_releases_prefill = (
        _cases.test_decode_abort_before_allocation_releases_prefill
    )
    test_retract_resume_recomputes_before_reallocation = (
        _cases.test_retract_resume_recomputes_before_reallocation
    )


class TestPrefillCompletePipelineParallelPrefill(
    _AllocationCases, _PipelineParallelPrefill, single_gpu._PrefillCompleteServer
):
    pass


class TestPrefillCompleteDpAttentionPrefill(
    _AllocationCases, _DpAttentionPrefill, single_gpu._PrefillCompleteServer
):
    pass


class TestJustInTimePipelineParallelPrefill(
    _AllocationCases, _PipelineParallelPrefill, single_gpu._PrefillCompleteServer
):
    policy = "just_in_time"


class TestJustInTimeDpAttentionPrefill(
    _AllocationCases, _DpAttentionPrefill, single_gpu._PrefillCompleteServer
):
    policy = "just_in_time"


class _MidPrefillDestination:
    """Alone on its server: with no handshake measured yet, readiness goes out
    at the request's first chunk, so decode's destination lands mid-prefill.
    After earlier requests it goes out about one handshake before the end."""

    policy = "just_in_time"

    _cases = single_gpu.TestJustInTimeAllocation
    test_destination_lands_mid_prefill_and_output_matches = (
        _cases.test_destination_lands_mid_prefill_and_output_matches
    )


class TestJustInTimeMidPrefillPipelineParallelPrefill(
    _MidPrefillDestination, _PipelineParallelPrefill, single_gpu._PrefillCompleteServer
):
    pass


class TestJustInTimeMidPrefillDpAttentionPrefill(
    _MidPrefillDestination, _DpAttentionPrefill, single_gpu._PrefillCompleteServer
):
    pass


if __name__ == "__main__":
    unittest.main()
