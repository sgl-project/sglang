"""Deterministic inference for GLM-4.7-Flash (Glm4MoeLiteForCausalLM) on Blackwell.

Two entry paths matter. The fa4 backend is what this model needs for
deterministic absorbed MLA, and the unspecified-backend path has to reach
triton through the absorbed-MLA arch probe instead of the flashinfer default
that every non-absorbed model gets on Blackwell.

Usage:
python3 -m unittest test_glm4_moe_lite_deterministic.TestGlm4MoeLiteFa4Deterministic
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_cuda_ci(est_time=360, stage="nightly", runner_config="4-gpu-b200")

GLM_MODEL = "zai-org/GLM-4.7-Flash"

# COMMON_SERVER_ARGS is shared module state; copy it. Extending it in place
# would leak the fa4 flag into the auto class below and silently make that
# test a second fa4 test.
SERVER_ARGS = COMMON_SERVER_ARGS + [
    "--chunked-prefill-size",
    "2048",
    "--max-prefill-tokens",
    "2048",
    "--mem-fraction-static",
    "0.8",
]


class TestGlm4MoeLiteFa4Deterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return GLM_MODEL

    @classmethod
    def get_server_args(cls):
        return SERVER_ARGS + ["--attention-backend", "fa4"]


class TestGlm4MoeLiteAutoBackendDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return GLM_MODEL

    @classmethod
    def get_server_args(cls):
        return SERVER_ARGS

    def test_auto_backend_resolves_to_triton(self):
        # Guards the arch probe itself: if Glm4MoeLiteForCausalLM stopped
        # counting as absorbed MLA the fill would hand back flashinfer, which
        # deterministic inference then rejects at launch.
        info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info").json()
        self.assertEqual(info["attention_backend"], "triton")


if __name__ == "__main__":
    unittest.main()
