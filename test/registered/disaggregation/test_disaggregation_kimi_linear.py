import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.server_fixtures.kimi_linear_pd_fixture import (
    KimiLinearPDParityMixin,
)

register_cuda_ci(est_time=480, stage="base-c", runner_config="4-gpu-h100")

DETERMINISTIC_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-deterministic-inference",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestKimiLinearHeterogeneousTPDisaggregation(
    KimiLinearPDParityMixin, PDDisaggregationServerBase
):
    prefill_tp_size = 2
    decode_tp_size = 1
    decode_base_gpu_id = 2
    reference_parallel_args = ["--tp-size", "2"]
    baseline_args = DETERMINISTIC_ARGS
    extra_prefill_args = DETERMINISTIC_ARGS
    extra_decode_args = DETERMINISTIC_ARGS


class TestKimiLinearPipelineDisaggregation(TestKimiLinearHeterogeneousTPDisaggregation):
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 2
    reference_parallel_args = ["--tp-size", "1", "--pp-size", "2"]
    extra_prefill_args = DETERMINISTIC_ARGS + ["--pp-size", "2"]


if __name__ == "__main__":
    unittest.main()
