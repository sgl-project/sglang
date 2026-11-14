"""GSM8K on a dense, non-sliding-window model with the SHUFFLE 5D KV layout.

The 5D layout decodes through ``pa_decode_gluon``, which reads ``kv_indices`` as a
2-D page table. Sliding-window models always get that table because they force
unified attention; a dense model at ``page_size > 1`` only gets it when the decode
metadata follows the KV layout, so this is the configuration that guards it.
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

register_amd_ci(est_time=210, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestAiterVectorized5DDenseDecode(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.7

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._env_stack = contextlib.ExitStack()
        cls._env_stack.enter_context(envs.SGLANG_USE_AITER.override(True))
        cls._env_stack.enter_context(
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override("vectorized_5d")
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "aiter", "--page-size", "64"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()


if __name__ == "__main__":
    unittest.main()
