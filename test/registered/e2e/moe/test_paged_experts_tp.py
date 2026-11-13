"""Paged experts match an unpaged TP2 server, in captured and host-planned execution."""

import unittest

from test_paged_experts import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    _PagedMatchesUnpaged,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")


class TestPagedExpertsTp2(_PagedMatchesUnpaged, CustomTestCase):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE
    server_args = ("--tp-size", "2", "--moe-runner-backend", "triton")
    # Each service uses two GPUs: the reference occupies 0,1 and the paged one 2,3.
    paged_extra_args = ("--base-gpu-id", "2")


if __name__ == "__main__":
    unittest.main(verbosity=3)
