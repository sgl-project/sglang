"""Matching DCP peers transfer their local unified pages without relayout."""

import unittest

from test_disaggregation_unified_memory import (
    KIMI_LINEAR_MODEL,
    SERVER_ENV,
    UNIFIED_MEMORY_ARGS,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=240, stage="extra-b", runner_config="8-gpu-h200")


class TestUnifiedMatchingDCP(PDLogprobParityMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    reference_parallel_args = ["--tp-size", "4"]
    baseline_args = UNIFIED_MEMORY_ARGS + [
        "--dcp-size",
        "2",
        "--attention-backend",
        "flashinfer",
        "--page-size",
        "4",
        "--chunked-prefill-size",
        "64",
    ]
    extra_prefill_args = baseline_args
    extra_decode_args = baseline_args
    parity_prompts = [[1] + [100 + i % 1000 for i in range(n)] for n in (65, 256)]
    parity_cached_prefix_min_prompt_tokens = 64


if __name__ == "__main__":
    unittest.main()
