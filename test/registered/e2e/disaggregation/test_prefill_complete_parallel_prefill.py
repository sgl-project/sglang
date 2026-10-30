"""Deferred destination KV allocation with a prefill server spread over two
GPUs: pipeline parallelism, or attention DP with expert parallelism."""

import unittest

# A module import, so the single-GPU classes are not collected again here.
import test_prefill_complete_allocation as single_gpu

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=420, stage="extra-b", runner_config="4-gpu-h100")


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


if __name__ == "__main__":
    unittest.main()
