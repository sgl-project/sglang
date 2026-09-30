"""Unit tests for the Nano Nemotron VL processor registry."""

import unittest

from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNanoNemotronVLProcessor(CustomTestCase):
    def test_supports_nemotron_h_omni(self):
        self.assertIn(
            NemotronH_Omni_Reasoning_V3,
            NanoNemotronVLImageProcessor.models,
        )


if __name__ == "__main__":
    unittest.main()
