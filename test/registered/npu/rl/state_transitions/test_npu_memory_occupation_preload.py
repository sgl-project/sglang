"""Nightly-only coverage for the LD_PRELOAD memory-saver hook mode."""

import unittest

from test_npu_memory_occupation import (
    MEMORY_ARGS,
    MODEL,
    PROMPTS,
    _MemoryOccupationCases,
)

from sglang.test.ascend.test_npu_rl_utils import (
    generation_signature,
    npu_rl_env,
    npu_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=300, suite="nightly-test-1-npu-a3", nightly=True)

PRELOAD_ENV = {**npu_rl_env(), "TMS_HOOK_MODE": "preload"}


class TestNpuMemoryOccupationPreload(_MemoryOccupationCases, CustomTestCase):
    """Same cases with the LD_PRELOAD memory-saver mode."""

    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=MEMORY_ARGS, env=PRELOAD_ENV)
        cls.base_url = cls._server.__enter__()
        cls.golden = generation_signature(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
