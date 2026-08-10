"""Unit tests for SGLang environment variable descriptors."""

import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMegaMoeEnvironment(CustomTestCase):
    def test_max_tokens_per_rank_default_and_override(self):
        field = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(field.name, None)
            self.assertEqual(field.get(), 8192)

        with field.override(8320):
            self.assertEqual(field.get(), 8320)


if __name__ == "__main__":
    unittest.main()
