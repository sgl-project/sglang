"""
CI-registered wrapper for SM120 MQA fallback kernel unit tests.

These tests run on any CUDA GPU (not SM120-specific) since the fallback
kernels are pure PyTorch implementations.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_sm120_mqa_fallback import (  # noqa: F401
    TestContiguousMQALogits,
    TestDequantFP8,
    TestPagedMQALogits,
    TestScheduleMetadata,
)

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
