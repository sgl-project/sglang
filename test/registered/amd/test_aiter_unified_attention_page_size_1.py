"""GSM8K with AITER unified attention on a dense model at page size 1.

Sliding-window models build the unified-attention decode page table through the SWA
pool, and Qwen3-VL turns unified attention on with page size 16. A dense model at the
default page size 1 is the configuration where the CUDA-graph decode page table is
copied from ``req_to_token`` as is, so this is the configuration that guards it.
"""

import contextlib
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=210, suite="stage-b-test-1-gpu-small-amd")


class TestAiterUnifiedAttentionPageSize1(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.65

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._env_stack = contextlib.ExitStack()
        cls._env_stack.enter_context(envs.SGLANG_USE_AITER.override(True))
        cls._env_stack.enter_context(envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(True))
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "aiter", "--page-size", "1"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()


if __name__ == "__main__":
    unittest.main()
