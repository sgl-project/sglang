"""Unified hybrid-MLA PD with state scatter and gather across attention TP."""

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

register_cuda_ci(est_time=400, stage="extra-b", runner_config="8-gpu-h200")


class _UnifiedTPConfig:
    model = KIMI_LINEAR_MODEL
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    baseline_args = UNIFIED_MEMORY_ARGS + ["--chunked-prefill-size", "64"]
    extra_prefill_args = baseline_args
    extra_decode_args = baseline_args
    parity_prompts = [[1] + [100 + i % 1000 for i in range(n)] for n in (65, 256)]
    parity_cached_prefix_min_prompt_tokens = 64


class TestUnifiedTPScatter(
    _UnifiedTPConfig, PDLogprobParityMixin, PDDisaggregationServerBase
):
    prefill_tp_size = 2
    decode_tp_size = 4
    decode_base_gpu_id = 2
    # Match prefill arithmetic; changing TP can change the first sampled token.
    reference_parallel_args = ["--tp-size", "2"]


class TestUnifiedTPGather(
    _UnifiedTPConfig, PDLogprobParityMixin, PDDisaggregationServerBase
):
    prefill_tp_size = 4
    decode_tp_size = 2
    decode_base_gpu_id = 4
    reference_parallel_args = ["--tp-size", "4"]


if __name__ == "__main__":
    unittest.main()
